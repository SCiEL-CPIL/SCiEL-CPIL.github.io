import numpy as np
from scipy.optimize import minimize

def construct():
    N = 16
    
    def objective(x_flat):
        x = x_flat.reshape(N, 2)
        # Calculate pairwise squared distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal (distance 0)
        np.fill_diagonal(dist_sq, np.inf)
        
        d_min = np.min(dist_sq)
        d_max = np.max(dist_sq)
        
        # Avoid division by zero
        if d_min < 1e-12:
            return 1e10
            
        # Minimize the squared ratio (equivalent to minimizing the ratio)
        return (d_max / d_min)
        
    best_score = np.inf
    best_points = None
    
    # Use multiple random restarts to escape local minima
    # Powell's method is derivative-free and robust for this non-smooth objective
    for seed in [42, 123, 999, 7, 314]:
        np.random.seed(seed)
        # Initialize with a random cluster
        x0 = np.random.randn(N, 2) * 0.5
        x0_flat = x0.flatten()
        
        try:
            result = minimize(objective, x0_flat, method='Powell', options={'maxiter': 1000, 'maxfev': 5000})
            
            if result.fun < best_score:
                best_score = result.fun
                best_points = result.x.reshape(N, 2)
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to random points if optimization fails
        np.random.seed(0)
        return np.random.randn(N, 2)
        
    # Center the points to remove any drift
    centroid = np.mean(best_points, axis=0)
    return best_points - centroid
