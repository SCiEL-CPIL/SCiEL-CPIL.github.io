import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
SQRT3 = np.sqrt(3)
V2_Y = SQRT3 / 2.0

def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            for k in range(j + 1, n):
                p3 = points[k]
                area = triangle_area(p1, p2, p3)
                if area < min_area:
                    min_area = area
    return min_area

def is_inside_triangle(x, y):
    if y < -1e-10:
        return False
    if SQRT3 * x - y < -1e-10:
        return False
    if SQRT3 * (1 - x) - y < -1e-10:
        return False
    return True

def points_inside(points):
    for p in points:
        if not is_inside_triangle(p[0], p[1]):
            return False
    return True

def de_objective(flat_points):
    points = flat_points.reshape(N, 2)
    if not points_inside(points):
        return 1e6
    return -min_triangle_area(points)

def slsqp_objective(flat_points):
    points = flat_points.reshape(N, 2)
    return -min_triangle_area(points)

def make_constraints():
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: SQRT3*x[2*i] - x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: SQRT3*(1 - x[2*i]) - x[2*i+1]})
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    bounds = [(0.0, 1.0), (0.0, V2_Y)] * N
    
    # 1. Global Search with DE
    best_de_result = None
    try:
        de_res = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=80, 
            popsize=15, 
            tol=1e-6,
            polish=False,
            updating='deferred'
        )
        if de_res.fun < 1e5:
            best_de_result = de_res.x
    except Exception:
        pass

    if time.time() > deadline:
        if best_de_result is not None:
            return best_de_result.reshape(N, 2)
        return np.random.rand(N, 2) * np.array([1.0, V2_Y])

    # 2. Local Polish with SLSQP
    x0 = best_de_result if best_de_result is not None else np.random.rand(N, 2) * np.array([1.0, V2_Y])
    x0 = np.clip(x0, [0.0, 0.0], [1.0, V2_Y])
    
    constraints = make_constraints()
    
    best_points = x0.reshape(N, 2)
    best_score = -slsqp_objective(x0)
    
    attempts = 0
    max_attempts = 20 # Increase attempts if time permits
    
    while time.time() < deadline and attempts < max_attempts:
        if attempts > 0:
            # Perturb
            noise = np.random.normal(0, 0.005, size=2*N)
            x0 = np.clip(x0 + noise, [0.0, 0.0], [1.0, V2_Y])
        
        try:
            res = minimize(
                slsqp_objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.success:
                curr_score = -res.fun
                if curr_score > best_score:
                    best_score = curr_score
                    best_points = res.x.reshape(N, 2)
                    x0 = res.x
            else:
                # Try Nelder-Mead as a fallback for this specific start?
                # NM doesn't handle constraints well.
                pass
        except Exception:
            pass
        
        attempts += 1
        
    return best_points
