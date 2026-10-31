import numpy as np
import time
from scipy.optimize import minimize

N = 38

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    mask = ~np.eye(len(pos), dtype=bool)
    r2 = r2[mask]
    # Avoid division by zero
    r2 = np.where(r2 < 1e-12, 1e-12, r2)
    r6 = r2 ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6)
    return energy

def _lj_grad(pos):
    """Compute gradient of total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    r2 = np.where(r2 < 1e-12, 1e-12, r2)
    r6 = r2 ** 3
    r12 = r6 ** 2
    # Gradient of 4 * (r^-12 - r^-6) w.r.t r is 4 * (-12 r^-13 + 6 r^-7)
    # dE/dr_i = dE/dr * dr/dr_i = dE/dr * (2 * (r_i - r'_i) / r)
    # dE/dr = 4 * (-12/r^13 + 6/r^7)
    # coeff = 4 * (-12/r^13 + 6/r^7) / (2r) * 2 ? 
    # Let's derive:
    # E = 4 (r^-12 - r^-6)
    # dE/dx = 4 (-12 r^-13 dx/dr - 6 r^-7 (-1) dx/dr) ? No.
    # d(r^-n)/dx = -n r^-n-1 dx/dr = -n r^-n-1 (x-x')/r = -n (x-x') r^-n-2
    # dE/dx = 4 [ -12 (x-x') r^-14 + 6 (x-x') r^-8 ]
    #      = 4 (x-x') [ -12 r^-14 + 6 r^-8 ]
    #      = 4 (x-x') r^-14 [ -12 + 6 r^6 ]
    #      = 4 (x-x') r^-8 [ -12 r^-6 + 6 ]
    #      = 24 (x-x') r^-8 [ 1 - 2 r^-6 ]
    
    # Using r2:
    # r^-14 = r2^-7
    # r^-8 = r2^-4
    # coeff = 4 * (-12 * r2^-7 + 6 * r2^-4)
    coeff = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _make_fcc_positions():
    """Generate the 38 FCC truncated octahedron positions."""
    # Generate FCC points on a grid
    # FCC points: (i, j, k) * a/2 where i+j+k is even.
    # We want the 38 closest to origin.
    # Max coordinate needed: The 38th point is in the shell with i^2+j^2+k^2=6.
    # Max i,j,k is 2. So range -3 to 3 is safe.
    
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                if (i + j + k) % 2 == 0:
                    points.append([i, j, k])
    
    pts = np.array(points, dtype=float)
    
    # Distance squared from origin
    dists = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the first N points
    cluster = pts[sorted_indices[:N]].copy()
    
    # Scale to optimal LJ distance
    # In this grid, nearest neighbor distance is sqrt(2) (e.g. (1,1,0) to (0,0,0) is sqrt(2)).
    # Optimal LJ distance r_min = 2^(1/6).
    # So scale factor s such that s * sqrt(2) = 2^(1/6).
    # s = 2^(1/6) / sqrt(2) = 2^(1/6) * 2^(-1/2) = 2^(1/6 - 3/6) = 2^(-2/6) = 2^(-1/3).
    # Wait.
    # r_ij = s * |p_i - p_j|.
    # For NN, |p_i - p_j| = sqrt(2).
    # We want r_ij = 2^(1/6).
    # s * sqrt(2) = 2^(1/6) => s = 2^(1/6) / 2^(1/2) = 2^(1/6 - 3/6) = 2^(-2/6) = 2^(-1/3).
    
    scale = 2.0 ** (-1.0 / 3.0)
    cluster *= scale
    
    # Center of mass to origin (should already be 0 due to symmetry, but just in case)
    cluster -= np.mean(cluster, axis=0)
    
    return cluster

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Generate the ideal FCC structure
    base_pos = _make_fcc_positions()
    
    # 2. Relax the structure
    # We will perform multiple relaxations with slight perturbations to ensure robustness
    best_pos = base_pos.copy()
    best_energy = _lj_energy(base_pos)
    
    # Define a relaxation function
    def relax(pos, max_iter=1000, tol=1e-12):
        try:
            res = minimize(
                _lj_energy,
                pos,
                jac=_lj_grad,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
            )
            return res.x
        except Exception:
            return pos

    # Initial relaxation of the ideal structure
    relaxed_pos = relax(base_pos)
    energy = _lj_energy(relaxed_pos)
    if energy < best_energy:
        best_energy = energy
        best_pos = relaxed_pos
    
    # 3. Multi-start with small perturbations
    # This helps if the ideal structure has slight numerical asymmetries or if the optimizer gets stuck in a very shallow local minimum
    rng = np.random.RandomState(42)
    
    for attempt in range(10):
        if time.time() > deadline - 20.0:
            break
        
        # Perturb the base structure
        # Add small noise to break symmetry and explore nearby minima
        noise = rng.normal(0, 0.01, size=(N, 3))
        perturbed_pos = base_pos + noise
        
        # Relax
        relaxed_pos = relax(perturbed_pos, max_iter=500, tol=1e-10)
        energy = _lj_energy(relaxed_pos)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed_pos
    
    # 4. Final high-precision polish
    if time.time() < deadline - 5.0:
        polished = relax(best_pos, max_iter=1000, tol=1e-14)
        energy = _lj_energy(polished)
        if energy < best_energy:
            best_energy = energy
            best_pos = polished
            
    return best_pos
