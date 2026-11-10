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
    deadline = time.time() + 590  # Leave 10s margin for the 600s limit

    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio ** 2

    def smooth_objective(pts_flat, beta=5.0):
        pts = pts_flat.reshape(n, d)
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        
        # Soft min and max
        # To avoid overflow, subtract max/min
        max_d = np.max(dists)
        min_d = np.min(dists)
        
        # log-sum-exp for max: log(sum(exp(beta*d))) = beta*max_d + log(sum(exp(beta*(d-max_d))))
        exp_max = np.exp(beta * (dists - max_d))
        soft_max = max_d + np.log(np.sum(exp_max)) / beta
        
        # log-sum-exp for min: log(sum(exp(-beta*d))) = -beta*min_d + log(sum(exp(-beta*(d-min_d))))
        # soft_min = - ( -beta*min_d + log(sum(exp(-beta*(d-min_d)))) ) / beta
        # soft_min = min_d - np.log(np.sum(np.exp(-beta * (dists - min_d)))) / beta
        exp_min = np.exp(-beta * (dists - min_d))
        soft_min = min_d - np.log(np.sum(exp_min)) / beta
        
        # We want to maximize soft_min / soft_max or soft_min - soft_max
        # Difference is smoother
        return -(soft_min - soft_max)

    def objective(pts_flat):
        return smooth_objective(pts_flat)

    # Initialize best result
    best_score = -1.0
    best_points = np.zeros((n, d))

    def init_points_on_sphere(seed):
        rng = np.random.RandomState(seed)
        points = rng.randn(n, d)
        norms = np.linalg.norm(points, axis=1)
        points /= norms[:, np.newaxis]
        return points

    # Multi-start L-BFGS-B
    num_starts = 20
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        if i == 0:
            # Fibonacci sphere
            points = np.zeros((n, d))
            golden_ratio = (1 + np.sqrt(5)) / 2
            for j in range(n):
                z = 1 - 2 * j / (n - 1)
                r = np.sqrt(1 - z**2)
                theta = 2 * np.pi * j / golden_ratio
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points[j] = [x, y, z]
        else:
            points = init_points_on_sphere(i)
            points += 0.05 * np.random.randn(n, d)

        pts_flat = points.flatten()

        options = {
            'maxiter': 5000,
            'ftol': 1e-12,
            'gtol': 1e-12
        }

        try:
            res = minimize(objective, pts_flat, method='L-BFGS-B', options=options)
            res_points = res.x.reshape(n, d)
            
            # Evaluate the actual metric on the result
            res_score = get_score(res_points)
            
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                
        except Exception:
            continue

    if best_score < 0:
        np.random.seed(42)
        best_points = np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
