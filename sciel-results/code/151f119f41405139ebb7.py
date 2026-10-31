import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2

def bary_to_cart(s, t):
    # s, t are arrays of shape (N,)
    # v = s, w = t * (1 - s)
    v = s
    w = t * (1 - s)
    x = v + 0.5 * w
    y = H * w
    return np.column_stack((x, y))

def min_triangle_area(points):
    # points: (N, 2)
    n = len(points)
    min_area = np.inf
    # Vectorized calculation of all triangle areas
    # Indices
    i, j, k = np.indices((n, n, n))
    # Mask for i < j < k
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Area = 0.5 * |det(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def objective(params):
    # params: shape (2*N,)
    s = params[0:N]
    t = params[N:2*N]
    points = bary_to_cart(s, t)
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 20s margin
    
    # Bounds
    bounds = [(0, 1)] * (2 * N)
    
    # Initial best
    best_points = None
    best_val = -np.inf
    
    # Run Differential Evolution
    try:
        # Use a fixed seed for reproducibility? No, random is better for exploration, 
        # but the task implies we should return a specific array. 
        # The seed program used a fixed seed. 
        # I will use a fixed seed to ensure deterministic output, 
        # but DE is stochastic. 
        # Let's fix the seed for the optimizer.
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=500,
            popsize=20,
            tol=1e-6,
            disp=False
        )
        
        if res.success or res.fun < best_val:
            best_val = -res.fun
            best_params = res.x
            best_points = bary_to_cart(best_params[:N], best_params[N:])
            
    except Exception as e:
        # Fallback to random if DE fails
        pass

    # If time allows, do local polish
    if best_points is not None:
        current_time = time.time()
        if current_time < deadline - 10:
            try:
                # Local optimization
                # Flatten points to 2N coords? 
                # It's easier to optimize in barycentric s,t space.
                res_local = minimize(
                    objective, 
                    best_params, 
                    method='Powell', 
                    bounds=bounds,
                    options={'maxiter': 1000, 'disp': False}
                )
                if res_local.fun < best_val:
                    best_val = -res_local.fun
                    best_params = res_local.x
                    best_points = bary_to_cart(best_params[:N], best_params[N:])
            except:
                pass
    
    if best_points is None:
        # Fallback: Random points
        rng = np.random.RandomState(0)
        s = rng.random(N)
        t = rng.random(N)
        best_points = bary_to_cart(s, t)
        
    return best_points
