"""Lennard-Jones cluster global minimum, 38 atoms.

Strategy: Build the FCC truncated octahedron (known global minimum topology)
as the primary seed, then use basin hopping with L-BFGS relaxation to refine
and potentially improve. Include a deadline guard.
"""
import time
import numpy as np
from itertools import combinations

N = 38
DEADLINE = 150.0  # wall-clock limit
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

# Precompute the FCC truncated octahedron structure for 38 atoms
# FCC lattice: conventional cell with atoms at corners and face centers
# Truncated octahedron: 13 inner (1 center + 12 first shell) + 25 outer (second shell)
# Actually for LJ38, the structure is:
# - 1 center atom
# - 12 atoms in first coordination shell (octahedral + tetrahedral sites)
# - 25 atoms in second coordination shell
# Total: 1 + 12 + 25 = 38

def _fcc_truncated_octahedron(scale=1.0):
    """Generate FCC truncated octahedron coordinates for 38 atoms."""
    # FCC lattice vectors: a/2 * (0,1,1), a/2*(1,0,1), a/2*(1,1,0)
    # For LJ, the minimum distance is 2^(1/6) * sigma ≈ 1.12246
    # FCC nearest neighbor distance = a/sqrt(2), so a = sqrt(2) * d_min
    
    # Let's use the standard FCC truncated octahedron with nearest-neighbor distance = 2^(1/6)
    d_min = 2.0 ** (1.0 / 6.0)
    
    # FCC basis in a cube of side a:
    # Atoms at (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)
    # a = d_min * sqrt(2)
    a = d_min * np.sqrt(2.0)
    
    # Generate all FCC lattice points in a 3x3x3 supercell
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner
                points.append((i * a, j * a, k * a))
                # Face centers
                points.append((i * a, j * a + a / 2, k * a + a / 2))
                points.append((i * a + a / 2, j * a, k * a + a / 2))
                points.append((i * a + a / 2, j * a + a / 2, k * a))
    
    points = np.array(points)
    
    # Center the structure
    center = points.mean(axis=0)
    points -= center
    
    # Select 38 atoms closest to center
    distances = np.linalg.norm(points, axis=1)
    idx = np.argsort(distances)
    points = points[idx[:38]]
    
    # Scale to ensure nearest-neighbor distance is exactly 2^(1/6)
    # Find minimum distance
    from scipy.spatial.distance import pdist
    min_d = np.min(pdist(points))
    points *= (d_min / min_d)
    
    return points


def _lj_energy_and_grad(pos):
    """Compute Lennard-Jones energy and gradient."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Using full matrix, each pair counted twice
    E = 2.0 * np.sum(4.0 * (r12 - r6))  # factor 2 because i<j and j<i
    
    # Gradient: dE/dr_i = 4 * sum_{j!=i} (12*r_i^-13 - 6*r_i^-7) * (r_i - r_j)
    # In vector form:
    # dE/dr_i = 4 * sum_j (12*r_ij^-14 - 6*r_ij^-8) * (r_i - r_j)
    # where r_ij is the vector from j to i
    
    # Using the matrix form:
    # grad[i] = 4 * sum_j (12 * r2_ij^-7 - 6 * r2_ij^-4) * diff[i,j]
    # where diff[i,j] = pos[i] - pos[j]
    
    # r2^-7 = r6 * r2^-1... let's be careful
    # r^-12 = r2^-6, r^-6 = r2^-3
    # d(r^-12)/dr_i = -12 * r^-14 * (r_i - r_j) ... wait
    # Let r = |r_i - r_j|, r^2 = r2
    # d(r^-12)/dr2 = -6 * r^-14 = -6 * r2^-7
    # d(r^-6)/dr2 = -3 * r^-8 = -3 * r2^-4
    # dE/dr_i = 4 * sum_j [d(r^-12)/dr_i - d(r^-6)/dr_i]
    # = 4 * sum_j [(-12*r^-13 + 6*r^-7) * (r_i - r_j)/r]  ... this is getting messy
    
    # Simpler: use the standard formula
    # grad_i = 4 * sum_{j != i} (12 * r_ij^-14 - 6 * r_ij^-8) * (pos_i - pos_j)
    # where r_ij = |pos_i - pos_j|
    
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    
    coeff = 4.0 * (12.0 * r ** (-14) - 6.0 * r ** (-8))
    coeff = coeff * diff  # broadcast: coeff[i,j] * (pos[i] - pos[j])
    
    grad = np.sum(coeff, axis=1)
    
    return E, grad


def _lbfgs_relax(pos, maxiter=500, tol=1e-10):
    """L-BFGS relaxation with deadline check."""
    from scipy.optimize import minimize
    
    def fun(x):
        E, g = _lj_energy_and_grad(x)
        return E, g
    
    result = minimize(fun, pos, method='L-BFGS-B', jac=True,
                      options={'maxiter': maxiter, 'gtol': tol, 'ftol': 1e-12})
    return result.x, result.fun


def construct():
    start_time = time.time()
    
    # Primary seed: FCC truncated octahedron
    pos0 = _fcc_truncated_octahedron()
    
    # Initial relaxation
    pos_best, e_best = _lbfgs_relax(pos0, maxiter=1000, tol=1e-12)
    
    # Basin hopping to try to find even lower energy
    np.random.seed(42)
    best_pos = pos_best.copy()
    best_energy = e_best
    
    # Try multiple basin hopping steps
    for step in range(200):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Random perturbation
        delta = np.random.randn(N, 3) * 0.15
        pos_trial = pos_best + delta
        
        # Relax
        pos_relaxed, e_relaxed = _lbfgs_relax(pos_trial, maxiter=500, tol=1e-10)
        
        # Metropolis acceptance (T decreases)
        T = 0.5 * np.exp(-0.02 * step)
        if e_relaxed < best_energy or np.random.rand() < np.exp(-(e_relaxed - best_energy) / T):
            pos_best = pos_relaxed
            if e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = pos_relaxed.copy()
    
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 2.0:
        final_pos, final_e = _lbfgs_relax(best_pos, maxiter=2000, tol=1e-14)
        if final_e < best_energy:
            best_pos = final_pos
            best_energy = final_e
    
    return best_pos
