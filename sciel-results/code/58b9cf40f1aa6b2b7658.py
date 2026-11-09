import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

# Precompute indices for vectorized area calculation
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def soft_min_area_vec(points, T):
    """
    Computes the soft minimum of the triangle areas.
    soft_min(x) = -T * log(sum(exp(-x_i / T)))
    As T -> 0, this approaches min(x).
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Avoid numerical issues if T is very small or areas are 0
    if T < 1e-10:
        return np.min(areas)
        
    # Shift for stability
    max_area = np.max(areas)
    if max_area == 0:
        return 0.0
        
    # We want min, so we use -soft_max of -areas? 
    # Or just use the formula for soft min directly.
    # log-sum-exp trick for stability:
    # sum(exp(-x/T)) = exp(-min(x)/T) * sum(exp(-(x-min(x))/T))
    # soft_min = -T * ( -min(x)/T + log(sum(exp(-(x-min(x))/T))) )
    #          = min(x) - T * log(sum(exp(-(x-min(x))/T)))
    
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
        
    exp_terms = np.exp(-(areas - min_area) / T)
    sum_exp = np.sum(exp_terms)
    
    return min_area - T * np.log(sum_exp)

def params_to_points_symmetric(params):
    """
    Maps 11 parameters to 11 points in the triangle, enforcing reflection symmetry.
    params[0]: y-coordinate of the central point (x=0.5)
    params[1:]: x, u for 5 pairs. x in [0, 0.5], u in [0, 1]. y = u * sqrt(3) * x.
    """
    params = np.array(params)
    points = np.zeros((N, 2))
    
    # Central point
    points[0, 0] = 0.5
    points[0, 1] = params[0]
    
    # 5 pairs
    for i in range(5):
        x = params[1 + 2*i]
        u = params[2 + 2*i]
        y = u * SQRT3 * x
        
        # Pair 1: (x, y)
        points[1 + 2*i, 0] = x
        points[1 + 2*i, 1] = y
        
        # Pair 2: (1-x, y)
        points[2 + 2*i, 0] = 1.0 - x
        points[2 + 2*i, 1] = y
        
    return points

def objective_soft(params, T):
    """
    Objective function to minimize (soft-min of triangle areas) for symmetric configuration.
    """
    points = params_to_points_symmetric(params)
    return -soft_min_area_vec(points, T)

def objective_hard(params):
    """
    Objective function to minimize (negative of min triangle area) for symmetric configuration.
    """
    points = params_to_points_symmetric(params)
    return -min_triangle_area_vec(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    # Bounds for parameters
    # y_0 in [0, H]
    # x_i in [0, 0.5]
    # u_i in [0, 1]
    bounds = [(0.0, H)]
    for _ in range(5):
        bounds.append((0.0, 0.5))
        bounds.append((0.0, 1.0))
        
    # Initialize with a reasonable configuration
    # Try a few random starts and pick the best one for the hard objective
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Global Search / Initialization
    # Use Differential Evolution for a short time to get a good starting point
    # Or just use random starts if DE is too slow
    try:
        # Short DE run
        result_de = differential_evolution(
            objective_hard, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            seed=42, 
            tol=1e-5,
            polish=False
        )
        best_params = result_de.x
        best_score = -result_de.fun
    except Exception:
        # Fallback to random
        best_params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        best_score = -objective_hard(best_params)

    # Phase 2: Simulated Annealing with Soft-Min
    # We will decrease T and minimize the soft objective
    # This allows the solver to "see" multiple constraints and navigate away from them
    
    T_values = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    current_params = best_params.copy()
    
    for T in T_values:
        if time.time() > deadline - 10.0:
            break
            
        def obj_func(params):
            return objective_soft(params, T)
        
        try:
            # Use Nelder-Mead or Powell for local refinement
            # Nelder-Mead is derivative-free and robust
            res = minimize(
                obj_func, 
                current_params, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.fun < objective_soft(current_params, T):
                current_params = res.x
        except Exception:
            pass
            
    # Phase 3: Final Polish with Hard Objective
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                objective_hard, 
                current_params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if -res.fun > best_score:
                best_params = res.x
                best_score = -res.fun
        except Exception:
            pass

    # Ensure we return the best found
    if best_params is None:
        best_params = current_params
        
    points = params_to_points_symmetric(best_params)
    
    return points
