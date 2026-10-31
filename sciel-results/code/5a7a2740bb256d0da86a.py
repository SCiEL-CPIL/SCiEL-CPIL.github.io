"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 110s safety margin out of 120s

def _remaining_time():
    return _DEADLINE - time.time()


def lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for N atoms."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    
    r6 = r2**(-3)
    r12 = r6**2
    
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Using full matrix, each pair counted twice
    energy = 2.0 * np.sum(4.0 * (r12 - r6))
    # But diagonal is excluded by fill_diagonal, so we need to be careful
    # Actually with fill_diagonal(r2, 1.0), the diagonal contributes 4*(1-1)=0, so it's fine
    # But the off-diagonal pairs are counted twice (i,j and j,i), so:
    # energy = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # = 2 * 4 * sum_{i<j} (...)  ... no
    # Let me redo: full sum over all i!=j of 4*(r^-12 - r^-6) = 2 * sum_{i<j} 4*(r^-12 - r^-6)
    # So energy = 0.5 * 4 * sum_{all i!=j} (r^-12 - r^-6)
    # = 2.0 * np.sum(4.0*(r12-r6)) would be 8*sum, which is wrong.
    
    # Let's just compute directly:
    # E = sum_{i<j} 4*(r_ij^-12 - r_ij^-6)
    # The full matrix sum (excluding diagonal) gives 2*E
    full_sum = np.sum(4.0 * (r12 - r6))
    energy = 0.5 * full_sum
    
    # Gradient: dE/dr2_ij = 4*(-6*r2^-7 - 6*r2^-4) ... wait
    # E = 4*(r^-12 - r^-6), r^2 = r2
    # dE/dr2 = 4*(-6*r2^-7 - 6*r2^-4) = -24*(r2^-7 + r2^-4)
    # Actually: r^-12 = r2^-6, r^-6 = r2^-3
    # dE/dr2 = 4*(-6*r2^-7 - (-3)*r2^-4) = 4*(-6*r2^-7 + 3*r2^-4) = -24*r2^-7 + 12*r2^-4
    # Gradient wrt position i: sum_{j!=i} dE/dr2_ij * d(r2_ij)/dx_i
    # d(r2_ij)/dx_i = -2*(x_j - x_i) = 2*(x_i - x_j)
    # So grad_i = sum_{j!=i} (-24*r2^-7 + 12*r2^-4) * 2*(pos_i - pos_j)
    
    # Vectorized:
    # coeff[i,j] = -24*r2^-7 + 12*r2^-4 for i!=j
    r2_safe = np.maximum(r2, 1e-12)
    coeff = -24.0 * r2_safe**(-7) + 12.0 * r2_safe**(-4)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = 2 * sum_j coeff[i,j] * (pos_i - pos_j)
    # = 2 * (sum_j coeff[i,j]*pos_i - sum_j coeff[i,j]*pos_j)
    # = 2 * pos_i * sum_j coeff[i,j] - 2 * (coeff @ pos)
    row_sum = coeff.sum(axis=1, keepdims=True)
    grad = 2.0 * (row_sum * pos - coeff @ pos)
    
    return energy, grad


def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 110.0
    
    best_energy = np.inf
    best_pos = None
    
    # FCC truncated octahedron coordinates for N=38
    # The 38-atom FCC cluster has shells: 1 (center) + 12 (first shell) + 14 (second shell) + 11 (third shell)
    # Actually the standard FCC(38) structure:
    # Let's use known coordinates from literature or generate FCC lattice and truncate.
    
    def _fcc_truncated_octahedron(n_atoms):
        """Generate FCC lattice points and select the n_atoms closest to center."""
        a = 1.12  # spacing close to LJ minimum
        # FCC lattice vectors
        # Generate a large FCC lattice and pick closest points
        max_coord = 6
        points = []
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                for k in range(-max_coord, max_coord + 1):
                    # FCC: (i,j,k) where i+j+k is even, scaled
                    if (i + j + k) % 2 == 0:
                        x = i * a / 2
                        y = j * a / 2
                        z = k * a / 2
                        points.append((x, y, z))
        
        points = np.array(points)
        # Sort by distance from center
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)
        return points[idx[:n_atoms]]
    
    # Try multiple strategies
    rng = np.random.default_rng(42)
    
    # Strategy 1: Start from FCC truncated octahedron and relax
    fcc_pos = _fcc_truncated_octahedron(N)
    
    # Ensure minimum separation >= 0.5
    def _enforce_min_sep(pos, min_sep=0.5):
        """Ensure all pairs are at least min_sep apart."""
        for _ in range(100):
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            r2 = np.sum(diff**2, axis=-1)
            np.fill_diagonal(r2, 1.0)
            min_r2 = np.min(r2)
            if min_r2 >= min_sep**2 - 1e-10:
                break
            # Find violating pairs and push apart
            r2_safe = np.maximum(r2, 1e-12)
            r = np.sqrt(r2_safe)
            # Push apart violating pairs
            for i in range(pos.shape[0]):
                for j in range(i + 1, pos.shape[0]):
                    if r2[i, j] < min_sep**2 - 1e-10:
                        diff_vec = pos[i] - pos[j]
                        r_ij = np.sqrt(r2[i, j])
                        if r_ij > 1e-10:
                            push = (min_sep - r_ij) / r_ij
                            pos[i] += 0.5 * push * diff_vec
                            pos[j] -= 0.5 * push * diff_vec
                        else:
                            pos[i] += min_sep / 2
            pos = pos - pos.mean(axis=0)
        return pos
    
    fcc_pos = _enforce_min_sep(fcc_pos.copy())
    
    # Relax the FCC structure
    if _remaining_time() > 5:
        try:
            res = minimize(lj_energy_and_grad, fcc_pos.copy(), method='L-BFGS-B',
                          jac=True, options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10})
            if res.success or res.fun < best_energy:
                # Enforce min separation
                pos_check = _enforce_min_sep(res.x.copy())
                e_check, _ = lj_energy_and_grad(pos_check)
                if e_check < best_energy:
                    best_energy = e_check
                    best_pos = pos_check
        except Exception:
            pass
    
    # Strategy 2: Basin hopping with random perturbations
    if best_pos is None:
        best_pos = fcc_pos.copy()
    
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    # Basin hopping
    n_basin_steps = 0
    while _remaining_time() > 20 and n_basin_steps < 50:
        n_basin_steps += 1
        # Random perturbation
        step_size = 0.3
        perturbed = current_pos + rng.normal(0, step_size, size=current_pos.shape)
        perturbed = _enforce_min_sep(perturbed)
        
        # Local relaxation
        try:
            res = minimize(lj_energy_and_grad, perturbed, method='L-BFGS-B',
                          jac=True, options={'maxiter': 300, 'gtol': 1e-6, 'ftol': 1e-8})
            if res.fun < current_energy - 1e-10:
                pos_check = _enforce_min_sep(res.x.copy())
                e_check, _ = lj_energy_and_grad(pos_check)
                if e_check < current_energy:
                    current_energy = e_check
                    current_pos = pos_check
                    best_energy = current_energy
                    best_pos = current_pos.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if _remaining_time() > 10:
        try:
            res = minimize(lj_energy_and_grad, best_pos.copy(), method='L-BFGS-B',
                          jac=True, options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-12})
            pos_check = _enforce_min_sep(res.x.copy())
            e_check, _ = lj_energy_and_grad(pos_check)
            if e_check < best_energy:
                best_energy = e_check
                best_pos = pos_check
        except Exception:
            pass
    
    # Final check: ensure minimum separation
    best_pos = _enforce_min_sep(best_pos.copy())
    
    return best_pos
# EVOLVE-BLOCK-END
