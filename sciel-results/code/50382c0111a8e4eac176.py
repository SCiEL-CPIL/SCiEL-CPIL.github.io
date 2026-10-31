import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

# Precompute triangle indices
indices = list(combinations(range(N), 3))
indices_arr = np.array(indices)

def get_min_area(points):
    """
    points: (N, 2) array
    returns: min area of any triangle
    """
    # p1, p2, p3: (165, 2)
    p1 = points[indices_arr[:, 0]]
    p2 = points[indices_arr[:, 1]]
    p3 = points[indices_arr[:, 2]]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def is_inside(points):
    """
    Check if all points are inside the triangle.
    Triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    # Check y >= 0
    if np.any(points[:, 1] < -1e-9):
        return False
    # Check y <= sqrt(3) * x  (for x in [0, 0.5])
    # Actually, the lines are:
    # 1. y = 0  => y >= 0
    # 2. y = sqrt(3) x => sqrt(3) x - y >= 0
    # 3. y = -sqrt(3) (x - 1) => sqrt(3) x + y - sqrt(3) <= 0
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Line 2: sqrt(3)*x - y >= 0
    if np.any(np.sqrt(3)*x - y < -1e-9):
        return False
    # Line 3: sqrt(3)*x + y - sqrt(3) <= 0
    if np.any(np.sqrt(3)*x + y - np.sqrt(3) > 1e-9):
        return False
        
    return True

def objective(x):
    """
    x: flat array of 22 floats
    returns: -min_area if valid, else penalty
    """
    points = x.reshape((N, 2))
    if not is_inside(points):
        return 1.0 # Penalty (max possible area is 0.5 * area of big tri = 0.5 * sqrt(3)/4 ~ 0.216)
    area = get_min_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Bounds for x and y
    # x in [0, 1], y in [0, sqrt(3)/2]
    h = np.sqrt(3)/2
    bounds = [(0, 1), (0, h)] * N
    
    # Initial guess: Symmetric distribution
    # 1 center, 6 hexagon, 4 corners-ish?
    # Let's just use random for DE, it's global.
    # But a good initial guess helps DE converge faster.
    # Let's create a simple grid-like initial guess.
    
    # Run Differential Evolution
    # popsize=20, maxiter=100 -> 2000 evals.
    # Each eval is fast.
    
    best_result = None
    best_score = -1.0
    
    try:
        # Use seed for reproducibility
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15, 
            tol=1e-5,
            update_interval=1,
            init='sobol' # Better distribution than latin hypercube sometimes
        )
        
        if res.success or res.fun < 1.0: # 1.0 is penalty
            best_points = res.x.reshape((N, 2))
            best_score = -res.fun
            best_result = best_points
        else:
            # Fallback to best found in res?
            # res.x is the best found
            best_points = res.x.reshape((N, 2))
            if is_inside(best_points):
                 best_score = get_min_area(best_points)
                 best_result = best_points
                 
    except Exception as e:
        pass

    # If DE didn't find a valid point or we have time, try local refinement or more iterations
    if best_result is None:
        # Fallback: Random search with best of N
        rng = np.random.RandomState(42)
        best_rand = None
        best_rand_score = -1
        for _ in range(1000):
            # Sample in triangle
            # u, v in [0,1], if u+v <= 1
            u = rng.random(N)
            v = rng.random(N)
            mask = u + v <= 1
            # This is tricky to sample exactly N points uniformly in triangle.
            # Simpler: sample in square, reject if outside.
            # Or use barycentric.
            
            # Let's just use the seed program's method as fallback
            import random
            rng2 = random.Random(7)
            points = []
            while len(points) < N:
                u, v = rng2.random(), rng2.random()
                if u + v < 0.98 and u > 0.01 and v > 0.01:
                    points.append([u + 0.5 * v, h * v])
            p = np.array(points)
            s = get_min_area(p)
            if s > best_rand_score:
                best_rand_score = s
                best_rand = p
        best_result = best_rand
        best_score = best_rand_score

    # Check time
    elapsed = time.time() - start_time
    if elapsed > 100:
        return best_result

    # Local refinement using SLSQP or L-BFGS-B
    # Since the objective is non-smooth, L-BFGS-B might jump out.
    # Let's try a simple gradient-free local search or just return DE result.
    # DE is usually good enough for this size.
    
    return best_result
