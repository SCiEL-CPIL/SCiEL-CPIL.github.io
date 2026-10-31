import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 135.0
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_and_grad(positions):
    """Compute LJ energy and gradient analytically (vectorized, optimized)."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    energy = 2.0 * np.sum(inv_r12 - inv_r6)
    coeff = 24.0 * inv_r2 * (inv_r6 - 2.0 * inv_r12)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return energy, grad

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation (optimized)."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    energy = 2.0 * np.sum(inv_r12 - inv_r6)
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart (vectorized)."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= min_dist * min_dist

def _generate_fcc_atoms():
    """
    Generate the first 40 FCC lattice points sorted by distance.
    Scale: 1/sqrt(2) so NN distance is 1.
    """
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 4 # Enough to cover shells up to r^2=6 (coords 2,2,2)
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: p[3])
    return points

def _get_fcc_candidates():
    """
    Generate candidate FCC truncated octahedron structures.
    The 38th atom is one of 8 equivalent atoms in the outer shell.
    We generate all 8 variants.
    """
    fcc_points = _generate_fcc_atoms()
    
    # The first 31 atoms are unique shells (1+12+6+12 = 31).
    # The next 8 are the outer shell (r^2=6).
    # We need 38 atoms. So we take the first 31, and 7 out of the next 8.
    # Actually, let's just take the first 39 and drop one from the last shell.
    
    # Verify shell structure
    # r2=0: 1
    # r2=1: 12 (Total 13)
    # r2=2: 6 (Total 19)
    # r2=4: 12 (Total 31)
    # r2=6: 8 (Total 39)
    
    core = [p[:3] for p in fcc_points[:31]]
    outer_shell = [p[:3] for p in fcc_points[31:39]]
    
    candidates = []
    for i in range(8):
        # Drop the i-th atom from the outer shell
        selected_outer = [outer_shell[j] for j in range(8) if j != i]
        candidate = np.array(core + selected_outer, dtype=float)
        candidates.append(candidate)
        
    return candidates

def relax(positions, maxiter=300, gtol=1e-8):
    """Relax using L-BFGS-B with analytical gradient."""
    def energy_and_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g
    
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_and_grad_func, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=500, gtol=1e-10):
    """High-precision final polish pass with analytical gradient."""
    def energy_and_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g
    
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_and_grad_func, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-15})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def basin_hopping(positions, n_iter=20, step_size=0.3, maxiter=100, temperature=0.05, seed=123):
    """Perform basin hopping optimization with analytical gradient."""
    def energy_and_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy_and_grad(pos)
    
    x0 = positions.flatten()
    current_energy, _ = lj_energy_and_grad(x0.reshape(N, 3))
    best_energy = current_energy
    best_x = x0.copy()
    
    rng = np.random.default_rng(seed)
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 15:
            break
        
        # Perturb
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        
        # Relax
        try:
            result = minimize(energy_and_grad_func, perturbed, method='L-BFGS-B', jac=True,
                            options={'maxiter': maxiter, 'gtol': 1e-7, 'maxls': 50, 'ftol': 1e-12})
            new_x = result.x
            new_energy, _ = lj_energy_and_grad(new_x.reshape(N, 3))
        except Exception:
            continue
        
        # Metropolis acceptance
        if new_energy < current_energy:
            x0 = new_x
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            delta = new_energy - current_energy
            if delta < 0:
                accept = 1.0
            else:
                accept = math.exp(-delta / temperature)
            if rng.uniform() < accept:
                x0 = new_x
                current_energy = new_energy
    
    return best_x.reshape(N, 3), best_energy

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Generate FCC candidates
    fcc_candidates = _get_fcc_candidates()
    
    # Relax each FCC candidate
    relaxed_candidates = []
    for i, cand in enumerate(fcc_candidates):
        if time.time() - START_TIME > DEADLINE - 60:
            break
        if not check_validity(cand):
            continue
            
        try:
            relaxed = relax(cand, maxiter=200, gtol=1e-7)
            if check_validity(relaxed):
                energy = lj_energy_vectorized(relaxed)
                relaxed_candidates.append((energy, relaxed))
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass
    
    # Sort relaxed candidates by energy
    relaxed_candidates.sort(key=lambda x: x[0])
    
    # 2. Basin Hopping on the best candidates
    # Use the top 3 relaxed candidates as starting points for BH
    bh_starts = []
    for e, pos in relaxed_candidates[:3]:
        bh_starts.append(pos)
        
    if not bh_starts and best_positions is not None:
        bh_starts.append(best_positions)
        
    bh_configs = [
        {'n_iter': 30, 'step_size': 0.2, 'maxiter': 80, 'temperature': 0.02, 'seed': 123},
        {'n_iter': 20, 'step_size': 0.4, 'maxiter': 70, 'temperature': 0.05, 'seed': 456},
    ]
    
    for start_pos in bh_starts:
        if time.time() - START_TIME > DEADLINE - 30:
            break
        for cfg in bh_configs:
            if time.time() - START_TIME > DEADLINE - 20:
                break
            try:
                bh_result, bh_energy = basin_hopping(start_pos, **cfg)
                if check_validity(bh_result) and bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_result
            except Exception:
                pass

    # 3. Final Polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = polish(best_positions, maxiter=500, gtol=1e-10)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    # Fallback
    if best_positions is None:
        # Use the first FCC candidate if relaxation failed
        if fcc_candidates:
            best_positions = fcc_candidates[0]
        else:
            # Simple cubic fallback
            side = int(np.ceil(N ** (1/3)))
            pts = []
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        pts.append([i, j, k])
            pts = np.array(pts[:N], dtype=float)
            # Center and scale
            pts -= pts.mean(axis=0)
            pts *= 1.2
            best_positions = pts

    return best_positions
