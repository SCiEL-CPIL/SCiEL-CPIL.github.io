import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
DEADLINE = 110.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    # Mapping from s1, s2 in [0,1] to triangle coordinates
    def map_to_triangle(s1, s2):
        # v = s1, w = s2 * (1 - s1)
        # x = v + 0.5 * w
        # y = sqrt(3)/2 * w
        v = s1
        w = s2 * (1.0 - s1)
        x = v + 0.5 * w
        y = (3 ** 0.5) / 2.0 * w
        return x, y

    def get_points_from_params(params):
        # params is shape (22,)
        s1 = params[0::2]
        s2 = params[1::2]
        xs = np.zeros(N)
        ys = np.zeros(N)
        for i in range(N):
            xs[i], ys[i] = map_to_triangle(s1[i], s2[i])
        return np.column_stack((xs, ys))

    def calculate_min_area(points):
        # points is (11, 2)
        # Calculate area for all triplets
        # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
        # Vectorized approach:
        # We can use the fact that Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # Or use cross product magnitude.
        
        # For N=11, a simple loop over combinations is fast enough and less memory intensive than creating a huge array.
        min_area = float('inf')
        # Pre-extract x and y for speed
        xs = points[:, 0]
        ys = points[:, 1]
        
        # Iterate over all triplets
        for i in range(N-2):
            xi, yi = xs[i], ys[i]
            for j in range(i+1, N-1):
                xj, yj = xs[j], ys[j]
                for k in range(j+1, N):
                    xk, yk = xs[k], ys[k]
                    # 2 * Area = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
                    area2 = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if area2 < min_area:
                        min_area = area2
        return min_area / 2.0

    def objective(params):
        points = get_points_from_params(params)
        min_area = calculate_min_area(points)
        return -min_area  # Minimize negative area

    # Bounds for s1, s2 are [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)

    # Global Optimization: Differential Evolution
    # Population size and max iterations need to be tuned for time.
    # 22 variables.
    # Let's run DE for a while, then polish.
    
    best_result = None
    best_score = -float('inf')

    # We will run DE with a time limit wrapper if possible, or just estimate.
    # SciPy DE doesn't have a direct time limit, but we can check elapsed time in a callback?
    # No, DE callback is per iteration. We can break early if time runs out?
    # Actually, it's easier to just run a fixed number of iterations that fits in the time, 
    # or use a manual loop with a local optimizer.
    
    # Given the complexity, let's try a multi-start local search with a global initializer.
    # Or just run DE with moderate settings.
    # 22 vars, popsize=15, maxiter=100 might be fast.
    
    try:
        # Use a seed for reproducibility if needed, but random is fine
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15, 
            tol=1e-6, 
            polish=False  # Let DE polish with L-BFGS-B by default? 
                          # L-BFGS-B might fail on non-smooth min. 
                          # We should probably disable polish in DE and do our own.
        )
        
        best_result = res.x
        best_score = -res.fun
        
    except Exception as e:
        # Fallback to random if DE fails
        print(f"DE failed: {e}")
        pass

    # If DE didn't run or we want to improve, do local refinement
    if best_result is None:
        # Random start
        best_result = np.random.rand(2 * N)
        best_score = -objective(best_result)

    # Local Refinement using Powell (handles non-smooth)
    # We can try multiple local refinements from the DE result and some random points
    # to ensure we escape local minima.
    
    current_best_params = best_result
    current_best_val = -best_score
    
    # Time check
    if time.time() - start_time < DEADLINE - 10:
        # Try refining the DE result
        try:
            res_local = minimize(
                objective, 
                current_best_params, 
                method='Powell', 
                bounds=bounds, 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_local.fun < objective(current_best_params):
                current_best_params = res_local.x
                current_best_val = -res_local.fun
        except Exception as e:
            print(f"Local refinement failed: {e}")

    # If we have time, try a few random restarts with local optimization
    # This helps if DE got stuck in a bad basin.
    attempts = 0
    while time.time() - start_time < DEADLINE - 5:
        # Random point in the square
        rand_params = np.random.rand(2 * N)
        try:
            res_rand = minimize(
                objective, 
                rand_params, 
                method='Powell', 
                bounds=bounds, 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if -res_rand.fun > current_best_val:
                current_best_params = res_rand.x
                current_best_val = -res_rand.fun
        except Exception:
            pass
        attempts += 1
        if attempts > 50: # Safety cap
            break

    # Convert best params to points
    best_points = get_points_from_params(current_best_params)
    
    return best_points

# The function name must be construct()
# The return type is array of shape (n, 2)
