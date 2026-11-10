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
    start_time = time.time()
    timeout = 580.0  # Safety margin before 630s limit (evaluator kills at 630, task says 600s finish, keep margin)

    def objective(x):
        """
        Objective function to minimize: - (d_min / d_max)^2
        x: 1D array of length n*d
        """
        # Check time
        if time.time() - start_time > timeout:
            raise TimeoutError()

        points = x.reshape(n, d)
        
        # Normalize points to the unit sphere to remove radial degrees of freedom
        # This helps the optimizer focus on angular distribution
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Handle zero vectors
        norms[norms == 0] = 1.0
        points = points / norms
        
        # Calculate pairwise distances
        dists = pdist(points)
        
        if len(dists) == 0:
            return 0.0

        d_min = np.min(dists)
        d_max = np.max(dists)
        
        # Handle degenerate cases
        if d_max == 0 or d_min == 0:
            return 1e6  # Large penalty

        ratio = d_min / d_max
        return - (ratio ** 2)

    # Initial points: Fibonacci Sphere
    def fib_sphere(n):
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    # Try a few different initializations to avoid local minima
    best_points = None
    best_score = -1e9

    # 1. Fibonacci Sphere
    init_points = fib_sphere(n)
    
    # 2. Random on Sphere (multiple seeds)
    np.random.seed(42)
    initializations = [init_points]
    for seed in range(1, 20):
        pts = np.random.randn(n, d)
        # Normalize to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        initializations.append(pts)
        
    # Also add perturbed Fibonacci
    for i in range(5):
        perturbed = init_points + np.random.randn(n, d) * 0.1
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms
        initializations.append(perturbed)

    for current_init in initializations:
        if time.time() - start_time > timeout - 30:
            break
            
        x0 = current_init.flatten()
        
        # Use Nelder-Mead
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                              options={'xatol': 1e-8, 'fatol': 1e-8, 
                                       'maxiter': 10000, 'maxfev': 20000,
                                       'adaptive': True})
            
            if result.success or (time.time() - start_time < timeout):
                # Re-normalize the result to ensure it's on the sphere
                res_points = result.x.reshape(n, d)
                norms = np.linalg.norm(res_points, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                res_points = res_points / norms
                
                # Calculate actual score on the normalized points
                dists = pdist(res_points)
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max > 0 and d_min > 0:
                    score = (d_min / d_max) ** 2
                else:
                    score = 0.0

                if score > best_score:
                    best_score = score
                    best_points = res_points
        except Exception:
            pass

    # Fallback if optimization failed or timed out
    if best_points is None:
        best_points = init_points
        # Ensure fallback is on sphere
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms

    # Final sanity check: ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = init_points

    return best_points

# EVOLVE-BLOCK-END
