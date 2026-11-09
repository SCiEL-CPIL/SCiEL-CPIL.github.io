import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    """
    Return the centers of 50 equal circles inside the unit disk.
    Uses a multi-start Nelder-Mead optimization approach to maximize the minimum radius.
    """
    best_pts = None
    best_score = -np.inf
    
    num_restarts = 30
    
    for restart in range(num_restarts):
        # Random initial points inside the unit disk
        r_init = np.sqrt(np.random.uniform(0, 1, N))
        theta_init = np.random.uniform(0, 2 * np.pi, N)
        x0 = np.column_stack((r_init * np.cos(theta_init), r_init * np.sin(theta_init))).flatten()
        
        def objective(x):
            pts = x.reshape(N, 2)
            
            # Calculate distances to origin
            dists_to_origin = np.linalg.norm(pts, axis=1)
            
            # Penalty for being outside
            outside_mask = dists_to_origin > 1.0
            if np.any(outside_mask):
                penalty = np.sum((dists_to_origin[outside_mask] - 1.0)**2)
            else:
                penalty = 0.0
            
            # Calculate minimum pairwise distance
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_mat = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist_mat, np.inf)
            min_pairwise = np.min(dist_mat)
            
            # Calculate radius
            r = min(min_pairwise / 2.0, 1.0 - np.max(dists_to_origin))
            
            # We want to maximize r, so minimize -r
            return -r + penalty
        
        # Use Nelder-Mead
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        
        # Evaluate the result
        if res.x is not None:
            pts = res.x.reshape(N, 2)
            # Verify validity
            dists_to_origin = np.linalg.norm(pts, axis=1)
            if np.all(dists_to_origin <= 1.0):
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dist_mat = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dist_mat, np.inf)
                min_pairwise = np.min(dist_mat)
                r = min(min_pairwise / 2.0, 1.0 - np.max(dists_to_origin))
                
                if r > best_score:
                    best_score = r
                    best_pts = pts.copy()
                    
    return best_pts
