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

    def score(points):
        """Compute R = (d_min / d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_scale(pts):
        """Normalize points so that d_max = 1, removing scale invariance."""
        dists = pdist(pts, 'euclidean')
        d_max = np.max(dists)
        if d_max < 1e-10:
            return pts
        return pts / d_max

    def objective(params):
        """
        Objective function to minimize: -d_min / d_max
        We maximize d_min/d_max, so we minimize its negative.
        Scale-normalize to remove scale invariance.
        """
        pts = params.reshape((n, 2))
        pts = normalize_scale(pts)
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0
            
        ratio = d_min / d_max
        return -ratio

    def smooth_objective(params, beta=100.0):
        """
        Smooth surrogate using soft-min and soft-max via log-sum-exp.
        Scale-normalize to remove scale invariance.
        """
        pts = params.reshape((n, 2))
        pts = normalize_scale(pts)
        dists = pdist(pts, 'euclidean')
        
        # Soft min: -logsumexp(-beta * dists) / beta
        # Soft max: logsumexp(beta * dists) / beta
        
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

    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        # Ensure finite and valid
        if not np.all(np.isfinite(pts)):
            return
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    def refine(pts, label=""):
        """
        Multi-stage refinement:
        1. L-BFGS-B on smooth objective (fast, global-ish)
        2. L-BFGS-B on raw objective (precise)
        3. Powell on raw objective (robust local search)
        4. Final high-precision L-BFGS-B
        """
        if time.time() > deadline - 10:
            return

        current = pts.copy()

        # Stage 1: Smooth L-BFGS-B
        try:
            res1 = minimize(
                smooth_objective, 
                current.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res1.x is not None and np.all(np.isfinite(res1.x)):
                current = res1.x.reshape((n, 2))
        except Exception:
            pass

        if time.time() > deadline - 10:
            update_best(current)
            return

        # Stage 2: Raw L-BFGS-B
        try:
            res2 = minimize(
                objective, 
                current.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                current = res2.x.reshape((n, 2))
        except Exception:
            pass

        if time.time() > deadline - 10:
            update_best(current)
            return

        # Stage 3: Powell (Derivative-free, good for non-smooth)
        try:
            res3 = minimize(
                objective, 
                current.flatten(), 
                method='Powell', 
                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            if res3.x is not None and np.all(np.isfinite(res3.x)):
                current = res3.x.reshape((n, 2))
        except Exception:
            pass

        if time.time() > deadline - 10:
            update_best(current)
            return

        # Stage 4: Final high-precision L-BFGS-B
        try:
            res4 = minimize(
                objective, 
                current.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res4.x is not None and np.all(np.isfinite(res4.x)):
                current = res4.x.reshape((n, 2))
        except Exception:
            pass

        update_best(current)

    # 1. Grid Initialization
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    refine(grid_points, "grid")

    # 2. Hexagonal Initialization
    if time.time() < deadline - 30:
        hex_points = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0],
            [0.5, 0.866], [1.5, 0.866], [2.5, 0.866], [3.5, 0.866],
            [0, 1.732], [1, 1.732], [2, 1.732], [3, 1.732],
            [0.5, 2.598], [1.5, 2.598], [2.5, 2.598], [3.5, 2.598]
        ])
        refine(hex_points, "hex")

    # 3. Fibonacci-like initialization
    if time.time() < deadline - 30:
        fib_points = np.zeros((n, 2))
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            angle = 2 * np.pi * i / golden_ratio
            radius = 1.0 + 0.5 * (i / n)
            fib_points[i, 0] = radius * np.cos(angle)
            fib_points[i, 1] = radius * np.sin(angle)
        refine(fib_points, "fib")

    # 4. Random Restarts
    np.random.seed(42)
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        # Random points in a square centered at origin
        rand_pts = np.random.rand(n, 2) * 10 - 5
        refine(rand_pts, f"rand_{i}")

    # 5. Final Polish on best found
    if best_points is not None and time.time() < deadline - 5:
        # One more high-precision pass on the best candidate
        try:
            res_final = minimize(
                objective, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res_final.x is not None and np.all(np.isfinite(res_final.x)):
                cand = res_final.x.reshape((n, 2))
                update_best(cand)
        except Exception:
            pass

    if best_points is None:
        # Fallback to grid if something went wrong
        return grid_points

    return best_points

# EVOLVE-BLOCK-END
