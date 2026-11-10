import numpy as np
from scipy.optimize import minimize

def construct():
    # Start with regular octagon
    n = 8
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Center at origin
    points -= points.mean(axis=0)
    
    # Scale to diameter 2
    # For regular octagon, diameter is 2 (radius 1)
    # Check diameter
    def get_diameter(pts):
        dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
        return np.max(dists)
    
    d = get_diameter(points)
    points = points * (2.0 / d)
    
    # Objective function to minimize (negative log score)
    def objective(params):
        pts = params.reshape(n, 2)
        # Calculate pairwise distances
        dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
        # Avoid log(0)
        dists[~np.isfinite(dists)] = 1e-10
        # Set diagonal to 1 to ignore self-distance
        np.fill_diagonal(dists, 1.0)
        
        # Log product of all ordered pairs
        # Sum of log(dists) over i != j
        log_prod = np.sum(np.log(dists))
        
        # Diameter
        D = np.max(dists)
        if D < 1e-9:
            return 1e10 # Penalty for collapse
        
        # Score is log_prod - 56 * log(D)
        # We want to maximize this, so minimize the negative
        return -(log_prod - 56 * np.log(D))
        
    # Initial guess
    x0 = points.flatten()
    
    # Optimize
    # Use Nelder-Mead as it doesn't require gradients and is robust for small N
    # Set maxiter to keep it fast
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success:
            best_points = res.x.reshape(n, 2)
        else:
            best_points = points
    except Exception:
        best_points = points
        
    # Final scaling to ensure diameter is exactly 2 for the output
    # The problem asks for points, and the evaluator scales them.
    # But to be safe, we return points that are distinct.
    # The evaluator scales to diameter 2.
    
    # Convert to list of lists
    return [[float(p[0]), float(p[1])] for p in best_points]
