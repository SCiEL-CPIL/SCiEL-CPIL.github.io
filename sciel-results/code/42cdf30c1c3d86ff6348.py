import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    deadline = time.time() + 590  # 590s to leave 10s margin for overhead
    
    best_score = -1.0
    best_points = np.random.randn(n, d)
    
    def get_score(pts):
        """
        Calculate the metric: (d_min / d_max)^2
        """
        # Calculate pairwise distances
        # pdist returns condensed distance matrix
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
            
        if d_min == 0:
            return 0.0
            
        return (d_min / d_max) ** 2

    def objective(pts_flat):
        """
        Objective function for minimization.
        We want to minimize -log(d_min) + log(d_max) which is equivalent to minimizing log(d_max/d_min).
        """
        pts = pts_flat.reshape(n, d)
        
        # Check for non-finite values
        if not np.all(np.isfinite(pts)):
            return 1e10
            
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 1e10
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_min <= 1e-12 or d_max <= 1e-12:
            return 1e10
            
        # Minimize log(d_max) - log(d_min)
        return np.log(d_max) - np.log(d_min)

    # Initial configurations
    np.random.seed(42)
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        # Initialize points on the unit sphere
        # Random points on sphere
        vecs = np.random.randn(n, d)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        pts = vecs / norms
        
        # Optionally, we can try to improve the initial guess with a few steps of repulsion
        # But minimize should handle it.
        
        # Flatten for optimizer
        x0 = pts.flatten()
        
        # Run optimization
        try:
            res = minimize(objective, x0, method='BFGS', options={'maxiter': 200, 'gtol': 1e-8})
            if res.success:
                opt_pts = res.x.reshape(n, d)
                score = get_score(opt_pts)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts
        except Exception:
            pass
            
        restarts += 1
        
        # If we've done a few restarts, check time more frequently
        if restarts % 5 == 0 and time.time() > deadline - 30:
            break

    return best_points
