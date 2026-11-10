# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, leave margin
    
    # Objective function: maximize A_min / A_hull
    # We minimize the negative of this ratio.
    def objective(x):
        points = x.reshape(-1, 2)
        
        # Check for invalid points
        if not np.all(np.isfinite(points)):
            return 1e10
            
        # Calculate convex hull
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
        except Exception:
            # If hull fails (e.g., all points collinear), return high cost
            return 1e10
            
        if A_hull < 1e-10:
            return 1e10

        # Calculate minimum triangle area
        # There are C(13, 3) = 286 triangles
        # For efficiency, we can vectorize this or use a loop.
        # A loop in Python might be slow, but 286 iterations is small.
        # However, inside the optimizer, this is called many times.
        # Let's try to be efficient.
        
        min_area = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                    area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                    if area < min_area:
                        min_area = area
                        
        if min_area < 1e-10:
             return 1e10
             
        ratio = min_area / A_hull
        return -ratio

    # Initial guess: Regular 13-gon
    # Center at (0,0), radius 1
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    best_x = x0
    best_score = objective(x0)
    
    # Optimization settings
    # Nelder-Mead is robust for non-smooth functions
    options = {
        'maxiter': 5000,
        'xatol': 1e-6,
        'fatol': 1e-6,
        'adaptive': True
    }
    
    # Run optimization
    # We might want to try multiple starts if time permits, but let's start with one robust run.
    # Regular polygon is a very strong start for this problem.
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options=options)
        
        if result.success or result.fun < best_score:
            if result.fun < best_score:
                best_x = result.x
                best_score = result.fun
    except Exception:
        pass
        
    # Check time
    if time.time() < deadline - 60.0:
        # Try a second run with a slightly perturbed start or different parameters?
        # Or maybe Lloyd's algorithm refinement?
        # Let's try a simple coordinate descent or random perturbation polish if time allows.
        # But Nelder-Mead usually converges well.
        
        # Let's try one more run with a different initial configuration: 
        # Points on a circle but slightly perturbed, or a known good configuration?
        # For n=13, the optimal configuration is not a regular polygon but close.
        # Let's stick with the result from the regular polygon start as it's very likely near-optimal.
        
        # If we have lots of time, we could run a multi-start.
        # But 600s is generous. Nelder-Mead on 26 variables with 286 area calcs per eval.
        # 286 * 3 mults is trivial. 5000 iterations * 286 ops is ~1.4M ops. Very fast.
        # So we can afford more iterations or multiple starts.
        
        # Let's do 3 restarts with slightly different initial radii or perturbations.
        for i in range(3):
            if time.time() > deadline - 30.0:
                break
                
            # Perturb the regular polygon
            rng = np.random.default_rng(seed=42 + i)
            x_init = x0.copy()
            # Small random noise
            x_init += rng.normal(0, 0.01, x_init.shape)
            
            try:
                res = minimize(objective, x_init, method='Nelder-Mead', options=options)
                if res.fun < best_score:
                    best_x = res.x
                    best_score = res.fun
            except Exception:
                pass

    # Final polish: The Nelder-Mead result might be stuck in a local minimum or have slightly suboptimal precision.
    # We can try a finer Nelder-Mead run on the best point found.
    if time.time() < deadline - 10.0:
        try:
            options_fine = {
                'maxiter': 2000,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
            res_fine = minimize(objective, best_x, method='Nelder-Mead', options=options_fine)
            if res_fine.fun < best_score:
                best_x = res_fine.x
                best_score = res_fine.fun
        except Exception:
            pass

    points = best_x.reshape(-1, 2)
    
    # Ensure valid shape and finite values
    if points.shape != (13, 2) or not np.all(np.isfinite(points)):
        # Fallback to regular polygon
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return points

# EVOLVE-BLOCK-END
