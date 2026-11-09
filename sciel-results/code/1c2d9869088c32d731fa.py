# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    
    def objective(flat_points):
        points = flat_points.reshape(n, 2)
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return -ratio * ratio

    best_score = -np.inf
    best_points = None
    
    # Multi-start optimization
    n_restarts = 50
    
    # Initial guesses: random uniform in [-1, 1]
    for i in range(n_restarts):
        if i == 0:
            # Start with a circle for the first one
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            init_points = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            init_points = np.random.uniform(-1, 1, (n, 2))
        
        # Flatten for scipy
        x0 = init_points.flatten()
        
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(n, 2)
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to seed behavior if all fail
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
# EVOLVE-BLOCK-END
