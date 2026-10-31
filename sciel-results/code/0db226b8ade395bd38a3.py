"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

try:
    import science_chemistry
    HAS_RELAX = True
except ImportError:
    HAS_RELAX = False

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def lj_energy(pos):
    """Calculate Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    # Avoid division by zero (though atoms should be distinct)
    np.fill_diagonal(dist_sq, 1.0) 
    dist = np.sqrt(dist_sq)
    r6 = dist ** 6
    r12 = r6 ** 2
    # Sum over i < j
    # Create mask for upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    e_pairs = 4 * (r12[mask] - r6[mask])
    return np.sum(e_pairs)

def lj_grad(pos):
    """Calculate Lennard-Jones gradient."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    r6 = dist ** 6
    r12 = r6 ** 2
    
    # Force on i due to j: F_ij = 24 * (2*r12 - r6) / r^2 * (r_j - r_i) / r
    # Gradient dE/dr = -F
    # dE/dx_i = sum_j dE/dr_ij * dx_ij/dr_ij
    # dx_ij/dr_ij = (x_i - x_j) / r_ij
    
    # Vectorized calculation
    # Factor = 24 * (2*r12 - r6) / r^2
    # Note: r^2 = dist_sq
    factor = 24 * (2 * r12 - r6) / dist_sq
    # Gradient contribution from pair (i,j) to i is factor[i,j] * (pos[i] - pos[j])
    # But we sum over all j != i.
    # grad[i] = sum_{j != i} factor[i,j] * (pos[i] - pos[j])
    # This can be computed as:
    # grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    # Wait, diff[i,j] = pos[i] - pos[j].
    # So grad[i] = sum_j factor[i,j] * diff[i,j]
    # Since factor is symmetric and diff is antisymmetric, this works.
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    return grad

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Generate FCC Lattice Fragment
    # Target nearest neighbor distance ~ 1.12
    a = 1.12 * np.sqrt(2)
    
    points = []
    # Range -5 to 5 is sufficient for 38 atoms
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                points.append((i * a, j * a, k * a))
                points.append((i * a, (j + 0.5) * a, (k + 0.5) * a))
                points.append(((i + 0.5) * a, j * a, (k + 0.5) * a))
                points.append(((i + 0.5) * a, (j + 0.5) * a, k * a))
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Select 38 closest
    initial_pos = points[sorted_indices[:N]].copy()
    # Center
    initial_pos -= np.mean(initial_pos, axis=0)
    
    best_pos = initial_pos
    best_energy = lj_energy(initial_pos)
    
    # 2. Relax using provided tool if available
    if HAS_RELAX:
        try:
            relaxed = science_chemistry.relax(initial_pos)
            if relaxed is not None and len(relaxed) == N:
                e_relaxed = lj_energy(relaxed)
                if e_relaxed < best_energy:
                    best_energy = e_relaxed
                    best_pos = relaxed
        except Exception:
            pass
        
        if time.time() > deadline:
            return best_pos

    # 3. If SciPy is available, try a local polish or alternate start
    if HAS_SCIPY:
        # Try optimizing the current best
        try:
            res = minimize(
                lj_energy, 
                best_pos, 
                jac=lj_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_energy:
                best_pos = res.x
                best_energy = res.fun
        except Exception:
            pass
        
        # If time remains, try an Icosahedral start?
        # The prompt suggests FCC is the target. 
        # But "bypassing the assumption" is in the directive.
        # However, the FCC basin is known to be the global min.
        # Let's stick to FCC to ensure speed and reliability.
        
        if time.time() > deadline:
            return best_pos

    return best_pos
# EVOLVE-BLOCK-END
