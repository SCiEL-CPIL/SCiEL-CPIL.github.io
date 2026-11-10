# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Initial guess: 4x4 grid
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))

    def score(points):
        """Compute R = (d_min / d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(params):
        """
        Objective function to minimize: -d_min / d_max
        We maximize d_min/d_max, so we minimize its negative.
        """
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0
            
        ratio = d_min / d_max
        return -ratio

    def smooth_objective(params, beta=50.0):
        """
        Smooth surrogate using soft-min and soft-max via log-sum-exp.
        """
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        
        # Soft min: -logsumexp(-beta * dists) / beta
        # Soft max: logsumexp(beta * dists) / beta
        # To avoid overflow, we can shift, but for this problem scale is small.
        
        # d_min approx
        neg_dists = -beta * dists
        max_neg_dists = np.max(neg_dists)
        log_sum_exp_neg = max_neg_dists + np.log(np.sum(np.exp(neg_dists - max_neg_dists)))
        d_min_soft = -log_sum_exp_neg / beta
        
        # d_max approx
        pos_dists = beta * dists
        max_pos_dists = np.max(pos_dists)
        log_sum_exp_pos = max_pos_dists + np.log(np.sum(np.exp(pos_dists - max_pos_dists)))
        d_max_soft = log_sum_exp_pos / beta
        
        if d_max_soft < 1e-10:
            return 0.0
            
        ratio = d_min_soft / d_max_soft
        return -ratio

    best_points = points.copy()
    best_score = score(best_points)

    # Helper to refine a candidate
    def refine(pts):
        nonlocal best_points, best_score
        
        # First pass: Smooth objective with L-BFGS-B to get close to optimum
        try:
            res1 = minimize(
                smooth_objective, 
                pts.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            cand1 = res1.x.reshape((n, 2))
            
            if time.time() > deadline:
                return

            # Second pass: Exact objective with L-BFGS-B for high precision
            res2 = minimize(
                objective, 
                cand1.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            cand2 = res2.x.reshape((n, 2))
            
            if time.time() > deadline:
                return

            score_val = score(cand2)
            if score_val > best_score:
                best_points = cand2
                best_score = score_val
                
        except Exception:
            pass

    # Try local optimization from the grid start
    refine(points)

    # Try a hexagonal-like arrangement
    if time.time() < deadline - 60:
        hex_points = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0],
            [0.5, 0.866], [1.5, 0.866], [2.5, 0.866], [3.5, 0.866],
            [0, 1.732], [1, 1.732], [2, 1.732], [3, 1.732],
            [0.5, 2.598], [1.5, 2.598], [2.5, 2.598], [3.5, 2.598]
        ])
        refine(hex_points)

    # Random restarts
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline - 30:
            break
            
        # Random points in a square
        rand_pts = np.random.rand(n, 2) * 10 - 5
        refine(rand_pts)

    return best_points

# EVOLVE-BLOCK-END
