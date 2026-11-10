import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s limit
    
    def get_points(params):
        # params: r1, r2, theta1, theta2, y
        r1, r2, theta1, theta2, y = params
        
        # Square 1: centered at (0,0)
        # Vertices at angles theta1 + k*pi/2
        angles1 = theta1 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        # Square 2: centered at (0, y)
        # Vertices at angles theta2 + k*pi/2
        angles2 = theta2 + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 + r2 * np.sin(angles2)])
        
        # Interleave points to form the 8-point set
        # Order doesn't matter for the product, but let's just concatenate
        # The set is {p1[0], p2[0], p1[1], p2[1], p1[2], p2[2], p1[3], p2[3]}
        # Actually, the labeling doesn't matter for the pairwise distances.
        points = np.vstack([p1, p2])
        return points

    def objective(params):
        points = get_points(params)
        
        # Calculate all pairwise distances
        # n=8, 28 pairs
        dists = []
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(points[i] - points[j])
                dists.append(d)
        
        dists = np.array(dists)
        
        # Handle zero distances (degenerate points)
        if np.any(dists < 1e-10):
            return 0.0
            
        # Diameter
        D = np.max(dists)
        
        # Product of distances
        # log product = sum log dists
        log_prod = np.sum(np.log(dists))
        
        # Score is proportional to log_prod - 28 * log(D)
        # We want to maximize this.
        # Minimize negative of this.
        val = log_prod - 28.0 * np.log(D)
        
        return -val

    def objective_raw_score(params):
        """Returns the actual score Delta/8^8 for reporting, but we optimize log-val"""
        points = get_points(params)
        dists = []
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(points[i] - points[j])
                dists.append(d)
        dists = np.array(dists)
        if np.any(dists < 1e-10):
            return 0.0
        D = np.max(dists)
        # Scale to diameter 2
        scale = 2.0 / D
        scaled_dists = dists * scale
        prod = np.prod(scaled_dists)
        # Score = prod / 8^8
        return prod / (8**8)

    # Initial guess: Regular Octagon
    # Regular octagon can be seen as two squares with r1=r2=1, theta1=0, theta2=pi/8, y=0?
    # No, regular octagon vertices are at angles 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4.
    # My parameterization:
    # Square 1: 0, pi/2, pi, 3pi/2 (angles relative to theta1)
    # Square 2: pi/4, 3pi/4, 5pi/4, 7pi/4 (angles relative to theta2)
    # If theta1=0, Square 1 is at 0, 90, 180, 270.
    # If theta2=pi/4, Square 2 is at 45, 135, 225, 315.
    # This matches the regular octagon if r1=r2=1 and y=0.
    
    bounds = [
        (0.1, 2.0),   # r1
        (0.1, 2.0),   # r2
        (0.0, 2*np.pi), # theta1
        (0.0, 2*np.pi), # theta2
        (-1.0, 1.0)   # y
    ]
    
    # Global search
    best_result = None
    best_val = -np.inf
    
    try:
        # Use differential evolution
        # maxfees limited by time
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10,
            tol=1e-6,
            polish=False, # We will do our own polish
            init='sobol'
        )
        
        if de_result.success or de_result.fun < best_val:
            best_params = de_result.x
            best_val = de_result.fun
            
            # Local polish
            local_res = minimize(
                objective, 
                best_params, 
                method='BFGS', 
                options={'maxiter': 1000, 'gtol': 1e-8}
            )
            
            if local_res.fun < best_val:
                best_params = local_res.x
                best_val = local_res.fun
                
            # Check time
            if time.time() > deadline - 20:
                break
                
    except Exception:
        pass

    # If DE failed or didn't run, fallback to regular octagon
    if best_params is None:
        best_params = np.array([1.0, 1.0, 0.0, np.pi/8, 0.0])
        
    # Additional local polish with Nelder-Mead for robustness
    try:
        if time.time() < deadline - 5:
            nm_res = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if nm_res.fun < best_val:
                best_params = nm_res.x
                best_val = nm_res.fun
    except Exception:
        pass

    # Convert best_params to points
    points = get_points(best_params)
    
    # Final scaling to ensure diameter is exactly 2 (floating point precision)
    dists = []
    for i in range(8):
        for j in range(i+1, 8):
            d = np.linalg.norm(points[i] - points[j])
            dists.append(d)
    D = np.max(dists)
    if D > 1e-10:
        points = points * (2.0 / D)
        
    # Verify distinctness and bounds
    # |coord| < 10^6 is easily satisfied for diameter 2
    # Distinctness: if any distance is 0, return regular octagon
    min_dist = np.min(dists)
    if min_dist < 1e-8:
        # Fallback to regular octagon
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    return [[float(p[0]), float(p[1])] for p in points]

if __name__ == "__main__":
    construct()
