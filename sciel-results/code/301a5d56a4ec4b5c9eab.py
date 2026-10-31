import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
# Vertices of equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])

def bary_to_cart(u, v):
    # w = 1 - u - v
    return u * V[0] + v * V[1] + (1.0 - u - v) * V[2]

def get_areas_vectorized(points):
    # points shape (N, 2)
    # Pre-compute combinations of indices
    # This is done once globally for efficiency
    pass

# Pre-compute index combinations for speed
COMBOS = np.array(list(itertools.combinations(range(N), 3)))

def calculate_min_area(points):
    p1 = points[COMBOS[:, 0]]
    p2 = points[COMBOS[:, 1]]
    p3 = points[COMBOS[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas), areas

def objective(x, beta, penalty_weight):
    uv = x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    # Penalty for u+v > 1
    viol = np.maximum(0.0, u + v - 1.0)
    penalty = penalty_weight * np.sum(viol**2)
    
    # Calculate points
    # Vectorized barycentric to cartesian
    # points = u[:,None]*V[0] + v[:,None]*V[1] + (1-u-v)[:,None]*V[2]
    w = 1.0 - u - v
    points = u[:, None] * V[0] + v[:, None] * V[1] + w[:, None] * V[2]
    
    # Calculate areas
    p1 = points[COMBOS[:, 0]]
    p2 = points[COMBOS[:, 1]]
    p3 = points[COMBOS[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min
    x_vals = -beta * areas
    m = np.max(x_vals)
    if m > 700:
        return 1e10
    log_sum = m + np.log(np.sum(np.exp(x_vals - m)))
    softmin_val = -log_sum / beta
    
    return -softmin_val + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    best_score = 0.0
    best_x = None
    
    # Initial guesses
    inits = []
    
    # 1. Random inits
    rng = np.random.default_rng(42)
    for _ in range(15):
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        inits.append(np.concatenate([u, v]))
        
    # 2. Structured init (Grid)
    # 3x3 grid in barycentric coordinates
    # u in {0.1, 0.5, 0.9}, v in {0.1, 0.5, 0.9}
    # But we need u+v <= 1.
    # (0.1, 0.1) -> 0.2 OK
    # (0.9, 0.9) -> 1.8 FAIL
    # Let's just generate valid grid points
    grid_u = [0.05, 0.3, 0.5, 0.7, 0.95]
    grid_v = [0.05, 0.3, 0.5, 0.7, 0.95]
    candidates = []
    for u in grid_u:
        for v in grid_v:
            if u + v <= 1.0:
                candidates.append((u, v))
    
    # Select 11 points from candidates to form a good spread
    # Simple approach: take first 11
    if len(candidates) >= N:
        u_grid = np.array([c[0] for c in candidates[:N]])
        v_grid = np.array([c[1] for c in candidates[:N]])
        inits.append(np.concatenate([u_grid, v_grid]))
        
    # Add corners and midpoints?
    # Corners: (0,0), (1,0), (0,1)
    # Midpoints: (0.5, 0), (0.5, 0.5), (0, 0.5)
    # Centroid: (1/3, 1/3)
    
    special_u = [0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1/3, 1/3, 1/3, 0.2, 0.8]
    special_v = [0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1/3, 1/3, 1/3, 0.2, 0.2]
    # Check validity
    valid_special = [(u, v) for u, v in zip(special_u, special_v) if u+v <= 1.0]
    if len(valid_special) >= N:
        u_sp = np.array([c[0] for c in valid_special[:N]])
        v_sp = np.array([c[1] for c in valid_special[:N]])
        inits.append(np.concatenate([u_sp, v_sp]))

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Stages: (beta, penalty_weight, maxiter)
    stages = [
        (10.0, 100.0, 100),
        (50.0, 1000.0, 150),
        (200.0, 10000.0, 200),
        (1000.0, 100000.0, 300)
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        x_current = init.copy()
        
        for stage_idx, (beta, pw, maxiter) in enumerate(stages):
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    objective, 
                    x_current, 
                    args=(beta, pw), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < objective(x_current, beta, pw):
                    x_current = res.x
                elif res.success:
                    x_current = res.x
            except Exception:
                pass

        # Evaluate final min area
        uv = x_current.reshape(N, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        points = u[:, None] * V[0] + v[:, None] * V[1] + w[:, None] * V[2]
        
        min_area, _ = calculate_min_area(points)
        
        if min_area > best_score:
            best_score = min_area
            best_x = x_current
            
    # Final Polish on best_x
    if best_x is not None and time.time() < deadline - 5.0:
        # Run one more time with very high beta
        try:
            res = minimize(
                objective, 
                best_x, 
                args=(5000.0, 1000000.0), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < objective(best_x, 5000.0, 1000000.0):
                best_x = res.x
        except Exception:
            pass

    # Return best points
    if best_x is None:
        # Fallback
        u = np.random.rand(N)
        v = np.random.rand(N) * (1.0 - u)
        x_final = np.concatenate([u, v])
    else:
        x_final = best_x
        
    uv = x_final.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    points = u[:, None] * V[0] + v[:, None] * V[1] + w[:, None] * V[2]
    
    return points
