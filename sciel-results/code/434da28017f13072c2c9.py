"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin for 120s limit
    
    # Initial hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            x0 = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98
    
    def score(x):
        """Compute the minimum radius for given centers."""
        x = x.reshape(-1, 2)
        # Distance to boundary
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        return min(min_pairwise / 2.0, np.min(dist_boundary))
    
    best_x = x0.copy()
    best_score = score(best_x)
    
    # Local optimization using scipy
    def neg_score(x):
        return -score(x)
    
    # Use L-BFGS-B with bounds
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Multiple restarts with small perturbations
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x_current = best_x.copy()
        else:
            # Perturb the best solution
            x_current = best_x + np.random.randn(2 * N) * 0.01
            # Ensure within bounds
            x_current = np.clip(x_current, -0.99, 0.99)
        
        try:
            result = minimize(neg_score, x_current, method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < -best_score:
                best_x = result.x.copy()
                best_score = -result.fun
        except Exception:
            continue
    
    # Check if we're still within deadline
    if time.time() < deadline - 5:
        # Try a few more iterations with tighter tolerance
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            try:
                result = minimize(neg_score, best_x, method='L-BFGS-B',
                                 bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                if result.fun < -best_score:
                    best_x = result.x.copy()
                    best_score = -result.fun
            except Exception:
                break
    
    # Final validation: ensure all points are inside the unit circle
    best_x = best_x.reshape(-1, 2)
    norms = np.linalg.norm(best_x, axis=1)
    # Clip any points that might be slightly outside
    mask = norms > 1.0
    if np.any(mask):
        best_x[mask] *= (1.0 / norms[mask] - 0.001)[:, np.newaxis]
    
    return best_x
# EVOLVE-BLOCK-END
