# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds limit, leave margin

    def compute_score(pts):
        # Normalize to unit sphere to make problem scale-invariant
        # Actually, the ratio d_min/d_max is scale invariant.
        # But we need to be careful with the optimization.
        # Let's maximize (d_min / d_max).
        # Equivalent to maximizing log(d_min) - log(d_max).
        # Or maximizing d_min subject to d_max <= 1.
        
        # Let's use the formulation: maximize t such that d_ij >= t for all i,j and d_ij <= 1 for all i,j?
        # No, we can scale. Let's fix d_max = 1? No, that's a constraint.
        
        # Better: Maximize f(P) = d_min(P) / d_max(P).
        # This is equivalent to maximizing d_min(P) subject to d_max(P) <= 1.
        # Or minimizing d_max(P) subject to d_min(P) >= 1.
        
        # Let's use scipy.optimize.minimize with a custom objective.
        # Objective to minimize: -log(d_min) + log(d_max)
        # This is smooth if d_min, d_max > 0.
        
        dists = pdist(pts)
        if np.any(dists <= 0):
            return 1e10
        d_min = np.min(dists)
        d_max = np.max(dists)
        return -np.log(d_min) + np.log(d_max)

    def objective(x):
        pts = x.reshape(n, d)
        return compute_score(pts)

    best_score = -np.inf
    best_points = None

    # Initial points: Random on sphere, or regular simplex projection?
    # Let's try several initializations.
    
    # 1. Random points on sphere
    np.random.seed(42)
    
    # 2. Icosahedron + 4 points? 12 vertices of icosahedron are good. Add 2 more.
    # 3. Random
    
    # Let's run Nelder-Mead or L-BFGS-B from multiple starts.
    # L-BFGS-B is faster for high dimensions.
    # Gradient can be computed numerically or analytically.
    # Let's use numerical gradient with L-BFGS-B or BFGS.
    
    # To speed up, we can use the fact that the objective is smooth.
    
    # Let's try a few restarts.
    num_restarts = 10
    max_iter = 1000
    
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Different initializations
        if i == 0:
            # Random
            pts_init = np.random.randn(n, d)
        elif i == 1:
            # Icosahedron vertices + 2 random
            # Icosahedron vertices:
            phi = (1 + np.sqrt(5)) / 2
            ico = np.array([
                [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
            ])
            ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
            pts_init = np.vstack([ico, np.random.randn(2, d)])
        else:
            pts_init = np.random.randn(n, d)
            pts_init = pts_init / np.linalg.norm(pts_init, axis=1, keepdims=True)

        x_init = pts_init.flatten()
        
        # Use L-BFGS-B
        try:
            res = minimize(objective, x_init, method='L-BFGS-B', options={'maxiter': max_iter, 'maxfun': 10000})
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(n, d)
        except Exception:
            pass

    # If no points found, return random
    if best_points is None:
        best_points = np.random.randn(n, d)

    # Optional: Polish with higher precision
    # We have time left, do a final polish
    if time.time() < deadline - 10:
        try:
            x_init = best_points.flatten()
            res = minimize(objective, x_init, method='L-BFGS-B', options={'maxiter': 2000, 'maxfun': 50000, 'ftol': 1e-15, 'gtol': 1e-15, 'xtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(n, d)
        except Exception:
            pass

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
