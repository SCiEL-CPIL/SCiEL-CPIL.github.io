import numpy as np
from scipy.optimize import minimize

def construct():
    n = 16
    d = 2
    
    def objective(params):
        points = params.reshape(n, d)
        # Regularization to keep scale reasonable
        reg = 0.01 * np.sum(points**2)
        
        # Calculate pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        i_upper = np.triu_indices(n, k=1)
        pairwise_dists = dists[i_upper]
        
        min_dist = np.min(pairwise_dists)
        max_dist = np.max(pairwise_dists)
        
        if min_dist < 1e-6:
            return 1e10 + reg
        
        return (max_dist / min_dist) ** 2 + reg

    best_score = float('inf')
    best_points = None
    
    # Multiple restarts
    for i in range(30):
        np.random.seed(1000 + i)
        # Initialize on a grid or random? Random is better for SA-like exploration
        # Let's try initializing on a circle or grid to give a better start
        # A hexagonal lattice start might be good, but we are trying to avoid that approach.
        # Let's use random points in a unit square, then let optimizer work.
        x0 = np.random.uniform(-1, 1, (n * d))
        
        try:
            res = minimize(objective, x0, method='Powell', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(n, d)
        except Exception:
            continue
            
    if best_points is None:
        return np.random.uniform(-1, 1, (n, d))
        
    return best_points
