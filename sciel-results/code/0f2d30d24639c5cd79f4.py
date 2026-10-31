"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 16
D = 2

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # safety margin
    
    # Start with a 4x4 grid
    side = 4
    x0 = np.array(list(__import__('itertools').product(range(side), repeat=D)), dtype=float)
    # Normalize to unit scale for optimization stability
    x0 = x0 - x0.mean(axis=0)
    x0 = x0 / np.max(np.linalg.norm(x0, axis=1)) * 2.0  # scale to roughly [-2, 2]
    
    best_points = x0.copy()
    best_score = np.inf
    
    def score(points):
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Exclude self-distances
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        max_d = np.max(dists)
        if min_d < 1e-12:
            return np.inf
        return (max_d / min_d) ** 2
    
    best_score = score(best_points)
    
    # Use scipy's differential_evolution for global optimization
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return best_points
    
    # Parameterize as 16 points in 2D = 32 parameters
    def objective(params):
        if time.time() > deadline:
            return 1e10  # signal timeout
        pts = params.reshape(N, D)
        s = score(pts)
        return s
    
    # Bounds: keep points in a reasonable range
    bounds = [(-5, 5)] * (N * D)
    
    # Run differential evolution with limited budget
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,  # we'll do our own polish
                update_interval=5
            )
            if result.success or result.fun < best_score:
                candidate = result.x.reshape(N, D)
                s = score(candidate)
                if s < best_score:
                    best_points = candidate
                    best_score = s
        except Exception:
            pass
    
    # Polish with local optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            from scipy.optimize import minimize
            # Try a few local optimizations from different starting points
            for start in [best_points, x0]:
                if time.time() > deadline - 3:
                    break
                def obj_local(params):
                    pts = params.reshape(N, D)
                    return score(pts)
                
                res = minimize(
                    obj_local,
                    start.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )
                if res.fun < best_score:
                    candidate = res.x.reshape(N, D)
                    s = score(candidate)
                    if s < best_score:
                        best_points = candidate
                        best_score = s
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
