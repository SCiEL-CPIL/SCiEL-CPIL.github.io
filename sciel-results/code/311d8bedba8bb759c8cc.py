"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total LJ energy for positions of shape (N, 3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 * r6
    # Sum over i < j
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    return energy

def _generate_fcc_38(a):
    """Generate 38 atoms in FCC truncated octahedron with lattice parameter a."""
    # FCC lattice points: (i, j, k) where i+j+k is even, scaled by a/2
    # We want points near origin
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    pts = np.array([i, j, k], dtype=float) * (a / 2.0)
                    points.append(pts)
    points = np.array(points)
    # Sort by distance from origin
    dists = np.sum(points * points, axis=1)
    idx = np.argsort(dists)
    # Take the 38 closest, but exclude duplicate at origin (there's only one)
    selected = points[idx[:N]]
    # Center the structure
    selected = selected - np.mean(selected, axis=0)
    return selected

def _optimize_structure(init_pos, time_limit=100.0):
    """Optimize the structure using L-BFGS with time limit."""
    start_time = time.time()
    
    def energy_func(x):
        pos = x.reshape(-1, 3)
        # Check minimum distance constraint
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2, 1.0)
        if np.min(r2) < 0.25:  # 0.5^2
            return 1e10
        return _lj_energy(pos)
    
    def gradient_func(x):
        pos = x.reshape(-1, 3)
        n = pos.shape[0]
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2, 1.0)
        r2 = np.maximum(r2, 1e-12)
        r = np.sqrt(r2)
        # Force on atom i: sum_j 4*(12/r^13 - 6/r^7) * (r_i - r_j)/r
        # In terms of r2: 4*(12*r2^-7 - 6*r2^-4) * (r_i - r_j)
        coeff = 4.0 * (12.0 * r2 ** -7 - 6.0 * r2 ** -4)
        np.fill_diagonal(coeff, 0.0)
        forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        return -forces.flatten()
    
    x0 = init_pos.flatten()
    
    # Try multiple restarts with small perturbations
    best_pos = init_pos.copy()
    best_energy = _lj_energy(init_pos)
    
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
        if restart == 0:
            x0 = init_pos.flatten()
        else:
            # Small random perturbation
            perturbation = np.random.randn(N, 3) * 0.05
            x0 = (init_pos + perturbation).flatten()
        
        try:
            result = minimize(
                energy_func, x0, jac=gradient_func,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pos = result.x.reshape(-1, 3)
            e = _lj_energy(pos)
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        except Exception:
            pass
    
    return best_pos, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Generate FCC truncated octahedron
    # The optimal lattice parameter for LJ is around 2^(1/6) ≈ 1.1225
    a = 2.0 ** (1.0 / 6.0)
    
    init_pos = _generate_fcc_38(a)
    
    # Verify minimum distance
    diff = init_pos[:, np.newaxis, :] - init_pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    min_dist = np.sqrt(np.min(r2))
    
    if min_dist < 0.5:
        # Scale up slightly
        scale = 0.5 / min_dist * 1.01
        init_pos = init_pos * scale
    
    # Optimize
    best_pos, best_energy = _optimize_structure(init_pos, time_limit=100.0)
    
    # Final check: ensure minimum distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    min_dist = np.sqrt(np.min(r2))
    
    if min_dist < 0.5:
        # Fallback to initial
        best_pos = init_pos
        best_energy = _lj_energy(init_pos)
    
    # Check time
    if time.time() > deadline:
        return init_pos
    
    return best_pos
# EVOLVE-BLOCK-END
