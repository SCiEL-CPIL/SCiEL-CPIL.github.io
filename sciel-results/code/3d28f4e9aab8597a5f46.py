import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    # Deadline guard: 600s limit, use 550s for safety
    start_time = time.time()
    timeout = 550.0

    def objective(params):
        """
        Objective function to minimize: negative of the squared min/max distance ratio.
        params: flattened array of shape (n*d,)
        """
        if time.time() - start_time > timeout:
            # Return a large value to signal termination if needed, 
            # though DE handles this via maxiter usually. 
            # For minimize, we might just let it run or check inside.
            pass
        
        points = params.reshape(n, d)
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 0.0
        
        ratio = d_min / d_max
        return -(ratio ** 2)

    # Initial guess: Regular polygon on unit circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_params = initial_points.flatten()

    # Global Search using Differential Evolution
    # Bounds: [-1, 1] for each coordinate
    bounds = [(-1.0, 1.0)] * (n * d)
    
    best_params = initial_params
    best_score = objective(initial_params)
    
    try:
        # Differential Evolution
        # maxiter=2000, popsize=20 should be fine within time limits for 32 vars
        de_result = differential_evolution(
            objective, 
            bounds, 
            maxiter=1000, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=False # We will polish manually to control time and method
        )
        
        if de_result.fun < best_score:
            best_params = de_result.x
            best_score = de_result.fun
            
    except Exception as e:
        print(f"DE failed: {e}")

    # Check time before polish
    elapsed = time.time() - start_time
    if elapsed < timeout - 60: # Leave 60s for polish and return
        try:
            # Local Polish using Powell (derivative-free, good for non-smooth)
            res = minimize(
                objective, 
                best_params, 
                method='Powell', 
                options={'maxiter': 1000, 'maxfev': 20000}
            )
            
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
                
        except Exception as e:
            print(f"Polish failed: {e}")
    
    # Convert back to points
    final_points = best_params.reshape(n, d)
    
    # Verify shape
    if final_points.shape != (n, d):
        # Fallback to initial if something went wrong
        final_points = initial_points

    return final_points

# EVOLVE-BLOCK-END
