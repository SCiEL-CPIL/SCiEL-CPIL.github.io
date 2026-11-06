"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def score(c):
        """Compute the minimum radius for a given set of centres."""
        c = np.asarray(c)
        if c.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(c)):
            return 0.0
        # Check bounds
        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 0.0
        # Compute pairwise distances
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        # Distance to walls
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist / 2.0, min_wall_dist)

    def negative_score(c):
        """Negative of the score for minimization."""
        return -score(c)

    # Initial guess: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c0 = grid[:N].copy()
    
    best_c = c0.copy()
    best_r = score(c0)
    
    # Try multiple random restarts with local optimization
    np.random.seed(42)
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate a random initial configuration
        if restart == 0:
            c_init = c0.copy()
        else:
            # Random points in the cube, with some perturbation
            c_init = np.random.uniform(-0.5, 0.5, size=(N, 3))
        
        # Use L-BFGS-B to maximize the minimum radius
        try:
            result = minimize(
                negative_score,
                c_init,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < -best_r:
                best_r = -result.fun
                best_c = result.x.copy()
        except Exception:
            continue
    
    # Final polish: try to improve with more iterations on the best solution
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                negative_score,
                best_c,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < -best_r:
                best_r = -result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Ensure the result is valid and within bounds
    best_c = np.clip(best_c, -0.5, 0.5)
    final_r = score(best_c)
    if final_r <= 0:
        # Fall back to the grid
        return c0
    
    return best_c
# EVOLVE-BLOCK-END
