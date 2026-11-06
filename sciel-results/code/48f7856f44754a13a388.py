import numpy as np
from scipy.optimize import minimize
import time

N = 130

def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return points / norms

def _covering_radius(points, test_points):
    # points: (N, 3)
    # test_points: (M, 3)
    # Compute distances from each test point to each configuration point
    # Use broadcasting: (M, 1, 3) - (1, N, 3) -> (M, N, 3)
    # This might be memory intensive for large M and N.
    # Alternative: compute in chunks.
    
    # For N=130 and M=10000, M*N*3 = 3,900,000 floats, which is ~30 MB. Acceptable.
    diff = test_points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (M, N)
    min_dists = np.min(dists, axis=1)  # (M,)
    return np.max(min_dists)

def _objective(flat_points):
    points = _normalize(flat_points.reshape(N, 3))
    # Use a fixed set of test points for consistency
    # We can generate them once outside, but for simplicity here, we use a global
    # Or better, pass them as a closure.
    # Let's assume TEST_POINTS is defined globally.
    return _covering_radius(points, TEST_POINTS)

def construct():
    global TEST_POINTS
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to leave margin

    # Generate initial configuration
    points = _fibonacci_spiral(N)
    
    # Generate test points for covering radius approximation
    # Use a finer Fibonacci spiral
    n_test = 20000
    TEST_POINTS = _fibonacci_spiral(n_test)
    
    # Initial objective
    current_obj = _objective(points.flatten())
    best_points = points.copy()
    best_obj = current_obj
    
    # Optimization
    # We can try to optimize. L-BFGS is good for smooth functions.
    # The function is not smooth due to the max and min, but it's a common heuristic.
    
    # To improve performance, we might use a coarser set of test points for the main loop
    # and refine at the end.
    
    # Let's try a few iterations of L-BFGS
    # We need to be careful with the time.
    
    # Alternative: Use a simple repulsion-based optimization which is faster to evaluate
    # but might not converge as well.
    
    # Given the time constraint, let's try a hybrid:
    # 1. A few steps of L-BFGS with a moderate number of test points.
    # 2. If time permits, refine with more test points.
    
    # For now, let's stick with L-BFGS and the 20k test points.
    # If it's too slow, we reduce n_test.
    
    try:
        res = minimize(
            _objective,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 100, 'gtol': 1e-5}
        )
        
        if res.success or res.fun < best_obj:
            best_points = _normalize(res.x.reshape(N, 3))
            best_obj = res.fun
            
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # Check time
    if time.time() > deadline:
        return best_points
    
    # If we have time, refine with more test points
    if time.time() < deadline - 20:
        n_test_fine = 50000
        TEST_POINTS = _fibonacci_spiral(n_test_fine)
        current_obj = _objective(best_points.flatten())
        
        try:
            res2 = minimize(
                _objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 30, 'maxfun': 50, 'gtol': 1e-6}
            )
            
            if res2.success or res2.fun < current_obj:
                best_points = _normalize(res2.x.reshape(N, 3))
                best_obj = res2.fun
                
        except Exception as e:
            print(f"Refinement failed: {e}")
    
    return best_points
