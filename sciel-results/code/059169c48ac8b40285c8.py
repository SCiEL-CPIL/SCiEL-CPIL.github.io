import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# EVOLVE-BLOCK-START

def objective(x):
    """
    Calculate the score: min triangle area / convex hull area.
    x: flattened array of shape (2*N,)
    """
    points = x.reshape(-1, 2)
    try:
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
        
        min_area = float('inf')
        n = len(points)
        # Vectorized triangle area calculation is possible but N=14 is small
        for i in range(n-2):
            pi = points[i]
            for j in range(i+1, n-1):
                v1 = points[j] - pi
                for k in range(j+1, n):
                    v2 = points[k] - pi
                    cross = v1[0]*v2[1] - v1[1]*v2[0]
                    area = 0.5 * abs(cross)
                    if area < min_area:
                        min_area = area
        return min_area / hull_area
    except:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial guess: regular N-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    best_points = x0
    best_score = objective(x0.flatten())
    
    def neg_objective(x):
        return -objective(x)
    
    # Global optimization with Differential Evolution
    try:
        bounds = [(-1, 1)] * (2*N)
        def callback(xk, convergence):
            if time.time() > deadline - 5:
                raise TimeoutError()
                
        # Use a moderate popsize and maxiter
        result = scipy.optimize.differential_evolution(
            neg_objective, 
            bounds, 
            maxiter=100, 
            tol=1e-6, 
            callback=callback, 
            seed=42,
            popsize=10
        )
        if result.fun < -best_score:
            best_score = -result.fun
            best_points = result.x.reshape(-1, 2)
    except TimeoutError:
        pass
    except Exception:
        pass
        
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            result = scipy.optimize.minimize(
                neg_objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_points = result.x.reshape(-1, 2)
        except Exception:
            pass
            
    return best_points

# EVOLVE-BLOCK-END
