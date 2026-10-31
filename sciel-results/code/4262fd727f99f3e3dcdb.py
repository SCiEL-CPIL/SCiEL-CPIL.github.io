import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    
    def soft_min(x, lam):
        m = np.min(x)
        return m - (1.0/lam) * np.log(np.sum(np.exp(-lam * (x - m))))

    def soft_max(x, lam):
        m = np.max(x)
        return m + (1.0/lam) * np.log(np.sum(np.exp(lam * (x - m))))

    def objective(params, lam):
        pts = params.reshape(-1, 2)
        # Calculate pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        d_sq = dist_sq[mask]
        
        # Avoid log(0) or division by zero issues in soft min/max
        # d_sq should be > 0 for distinct points
        # If points coincide, d_sq has 0.
        # soft_min of 0 is 0.
        # If sm == 0, ratio is inf.
        
        sm = soft_min(d_sq, lam)
        smax = soft_max(d_sq, lam)
        
        if sm < 1e-12:
            return 1e10
        
        # Regularization to fix scale (prevent drift to 0 or inf)
        # We want typical distance ~ 1
        reg = 1e-6 * (np.mean(d_sq) - 1.0)**2
        
        return smax / sm + reg

    # Initial guess: 4x4 grid
    side = 4
    x = np.linspace(0, 1, side)
    y = np.linspace(0, 1, side)
    xx, yy = np.meshgrid(x, y)
    init_grid = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Center the grid to help with optimization
    init_grid = init_grid - np.mean(init_grid, axis=0)
    
    best_score = np.inf
    best_points = init_grid.copy()
    
    # Try different lambda values for soft approximation
    lams = [10, 50, 100, 200]
    
    # Use a fixed random seed for reproducibility of the "random" starts
    rng = np.random.default_rng(42)
    
    for lam in lams:
        if time.time() - start_time > 110:
            break
        
        # Try 5 different starts
        for i in range(5):
            if time.time() - start_time > 110:
                break
            
            if i == 0:
                pts0 = init_grid
            else:
                # Perturb the grid
                pts0 = init_grid + rng.normal(0, 0.05, (N, 2))
            
            # Run optimizer
            res = minimize(objective, pts0.ravel(), args=(lam,), method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15})
            
            if res.success:
                pts = res.x.reshape(N, 2)
                
                # Calculate true score
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dist_sq = np.sum(diff**2, axis=2)
                mask = ~np.eye(N, dtype=bool)
                d_sq = dist_sq[mask]
                
                d_min = np.min(d_sq)
                d_max = np.max(d_sq)
                
                if d_min < 1e-12:
                    continue
                    
                true_score = (d_max / d_min)
                # Task score is squared ratio. 
                # Minimizing ratio is equivalent to minimizing squared ratio.
                # So we can compare true_score directly.
                
                if true_score < best_score:
                    best_score = true_score
                    best_points = pts
    
    return best_points
