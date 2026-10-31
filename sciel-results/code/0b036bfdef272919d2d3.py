import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

# Triangle vertices
V0 = np.array([0, 0])
V1 = np.array([1, 0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    P = u*V0 + v*V1 + (1-u-v)*V2
    u: (N,)
    v: (N,)
    Returns: (N, 2)
    """
    w = 1.0 - u - v
    px = v + 0.5 * w
    py = H * w
    return np.column_stack((px, py))

def calculate_areas(points):
    """
    points: (N, 2) array (Cartesian)
    Returns: array of areas for all triangles
    """
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    areas = 0.5 * np.abs(cross)
    return areas

def soft_min_neg(areas, beta):
    """
    Returns -SoftMin(areas) using stable log-sum-exp
    """
    min_area = np.min(areas)
    # shifted = -beta * (areas - min_area)
    # log_sum = -beta * min_area + log(sum(exp(shifted)))
    shifted = -beta * (areas - min_area)
    # Clip shifted to avoid overflow in exp (though shifted <= 0, so exp <= 1)
    # But if beta is huge and areas-min_area is tiny, it's fine.
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    log_sum = -beta * min_area + log_sum_exp
    return log_sum / beta

def objective(params, beta):
    """
    params: (2N,) array of barycentric coordinates [u0, v0, u1, v1, ...]
    """
    u = params[0::2]
    v = params[1::2]
    points = bary_to_cart(u, v)
    areas = calculate_areas(points)
    return soft_min_neg(areas, beta)

def make_bary_constraints():
    """
    Constraints for barycentric coordinates:
    u_i + v_i <= 1  =>  1 - u_i - v_i >= 0
    """
    cons = []
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[2*i] - p[2*i+1]
        })
    return cons

CONSTRAINTS = make_bary_constraints()

def clamp_point_to_triangle(x, y):
    """
    Clamp a single point (x, y) to be inside the equilateral triangle.
    """
    # Edge 1: y >= 0
    if y < 1e-9:
        y = 1e-9
    # Edge 2: y <= sqrt(3) * x
    if y > np.sqrt(3) * x + 1e-9:
        # Project onto y = sqrt(3) * x
        # Line: sqrt(3)*x - y = 0
        # Normal: (sqrt(3), -1)
        # d = (sqrt(3)*x - y) / (3 + 1) = (sqrt(3)*x - y) / 4
        d = (np.sqrt(3) * x - y) / 4.0
        x = x - np.sqrt(3) * d
        y = y + d
    # Edge 3: y <= sqrt(3) * (1 - x)
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        # Line: sqrt(3)*x + y - sqrt(3) = 0
        # Normal: (sqrt(3), 1)
        # d = (sqrt(3)*x + y - sqrt(3)) / (3 + 1)
        d = (np.sqrt(3) * x + y - np.sqrt(3)) / 4.0
        x = x - np.sqrt(3) * d
        y = y - d
        
    # Final safety check: if still outside (due to numerical issues), move to centroid
    cx, cy = 0.5, H / 3.0
    if y < 0 or y > np.sqrt(3) * x or y > np.sqrt(3) * (1 - x):
        # Blend towards centroid
        x = 0.9 * x + 0.1 * cx
        y = 0.9 * y + 0.1 * cy
        # Re-clamp
        if y < 1e-9: y = 1e-9
        if y > np.sqrt(3) * x: y = np.sqrt(3) * x * 0.99
        if y > np.sqrt(3) * (1 - x): y = np.sqrt(3) * (1 - x) * 0.99
        
    return x, y

def clamp_points_to_triangle(points):
    """
    Clamp all points in (N, 2) array.
    """
    pts = points.copy()
    for i in range(N):
        pts[i, 0], pts[i, 1] = clamp_point_to_triangle(pts[i, 0], pts[i, 1])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Bounds for barycentric coords: u in [0, 1], v in [0, 1]
    bounds = [(0, 1), (0, 1)] * N
    
    best_points = None
    best_val = -1
    
    rng = np.random.default_rng(42)
    
    # Generate initial guesses in barycentric coords
    inits = []
    for _ in range(10):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        params = np.zeros(2 * N)
        params[0::2] = u
        params[1::2] = v
        inits.append(params)
        
    # Annealing betas
    betas = [20, 50, 100, 200, 500]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
            
        current = init.copy()
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            # Minimize soft_min_neg
            max_iter = 50 if beta < 100 else 100
            
            try:
                res = minimize(
                    objective, 
                    current, 
                    args=(beta,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=CONSTRAINTS,
                    options={'maxiter': max_iter, 'ftol': 1e-10}
                )
                
                # Check if result is valid
                if res.success:
                    u_res = res.x[0::2]
                    v_res = res.x[1::2]
                    # Verify feasibility
                    if np.all(u_res >= -1e-8) and np.all(v_res >= -1e-8) and np.all(u_res + v_res <= 1 + 1e-8):
                        # Clip to [0, 1]
                        u_res = np.clip(u_res, 0, 1)
                        v_res = np.clip(v_res, 0, 1)
                        new_params = np.zeros(2 * N)
                        new_params[0::2] = u_res
                        new_params[1::2] = v_res
                        current = new_params
            except Exception:
                continue

        # Evaluate final raw min area
        u_final = current[0::2]
        v_final = current[1::2]
        final_points = bary_to_cart(u_final, v_final)
        
        # Clamp to ensure strictly inside
        final_points = clamp_points_to_triangle(final_points)
        
        areas = calculate_areas(final_points)
        val = np.min(areas)
        
        if val > best_val:
            best_val = val
            best_points = final_points
            
    if best_points is None:
        # Fallback
        rng2 = np.random.default_rng(7)
        u = rng2.random(N)
        v = rng2.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        best_points = bary_to_cart(u, v)
        best_points = clamp_points_to_triangle(best_points)
        
    return best_points
