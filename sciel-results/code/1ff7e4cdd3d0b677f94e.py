import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 12

def hex_support(angle, orientation):
    """
    Calculate the support function of a unit regular hexagon 
    in the given direction angle, given the hexagon's orientation.
    """
    # Angle of normals in global frame:
    # normal_k = orientation + pi/6 + k*pi/3
    
    # We want the angle difference between 'angle' and the nearest normal.
    # Map to local frame relative to orientation
    rel_angle = (angle - orientation) % (np.pi/3)
    # The normals are at pi/6, 2pi/6, etc.
    # The angle from the nearest normal (pi/6) is:
    # If rel_angle < pi/6, diff = pi/6 - rel_angle
    # If rel_angle >= pi/6, diff = rel_angle - pi/6 (but wait, next normal is at 2pi/6)
    # Actually, the sector is [0, pi/3). The normal is at pi/6.
    # So delta = abs(rel_angle - pi/6)
    # This works because the function is symmetric around the normal.
    
    delta = abs(rel_angle - np.pi/6)
    
    return (np.sqrt(3) / 2) / np.cos(delta)

def overlap_penalty(params, i, j):
    x1, y1, t1 = params[i*3:(i+1)*3]
    x2, y2, t2 = params[j*3:(j+1)*3]
    
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx*dx + dy*dy)
    
    if dist < 1e-9:
        return 1000.0
    
    angle = np.arctan2(dy, dx)
    
    h1 = hex_support(angle, t1)
    h2 = hex_support(angle + np.pi, t2)
    
    req_dist = h1 + h2
    
    if dist < req_dist:
        return (req_dist - dist) ** 2
    return 0.0

def containment_penalty(params, S, outer_center, outer_rot):
    A = S * np.sqrt(3) / 2
    penalty = 0.0
    
    for i in range(N):
        x, y, t = params[i*3:(i+1)*3]
        
        for k in range(6):
            vx = x + np.cos(t + k*np.pi/3)
            vy = y + np.sin(t + k*np.pi/3)
            
            px = vx - outer_center[0]
            py = vy - outer_center[1]
            
            cos_r = np.cos(-outer_rot)
            sin_r = np.sin(-outer_rot)
            lx = px * cos_r - py * sin_r
            ly = px * sin_r + py * cos_r
            
            for m in range(6):
                n_angle = np.pi/6 + m*np.pi/3
                proj = lx * np.cos(n_angle) + ly * np.sin(n_angle)
                if proj > A:
                    penalty += (proj - A) ** 2
                    
    return penalty

def cost_function(flat_params):
    params = flat_params[:N*3]
    S = flat_params[N*3]
    outer_rot = flat_params[N*3 + 1]
    
    if S < 1.0:
        S = 1.0
    
    pen_overlap = 0.0
    for i in range(N):
        for j in range(i+1, N):
            pen_overlap += overlap_penalty(params, i, j)
            
    pen_contain = containment_penalty(params, S, (0.0, 0.0), outer_rot)
    
    return S + 100.0 * (pen_overlap + pen_contain)

def construct():
    start_time = time.time()
    
    bounds = []
    for _ in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, np.pi/3))
    bounds.append((3.0, 10.0))
    bounds.append((0.0, np.pi/3))
    
    def objective(x):
        return cost_function(x)
        
    best_result = None
    best_cost = float('inf')
    
    try:
        # Use DE to find a good basin
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=150, 
            popsize=25, 
            tol=1e-6, 
            seed=42,
            polish=False
        )
        
        best_result = result.x
        best_cost = result.fun
        
    except Exception:
        # Fallback
        inner = []
        for i in range(N):
            inner.append([i * np.sqrt(3), 0, 0])
        return inner, (0.0, 0.0), N * np.sqrt(3) + 2, 0.0

    elapsed = time.time() - start_time
    
    # Refine with Powell (handles non-smoothness better than L-BFGS-B)
    if elapsed < 100:
        try:
            result_refined = minimize(
                objective, 
                best_result, 
                method='Powell', 
                bounds=bounds,
                options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            if result_refined.fun < best_cost:
                best_result = result_refined.x
                best_cost = result_refined.fun
        except Exception:
            pass
            
    params = best_result[:N*3]
    S = best_result[N*3]
    outer_rot = best_result[N*3 + 1]
    
    inner = []
    for i in range(N):
        x, y, t = params[i*3:(i+1)*3]
        inner.append([x, y, t * 180.0 / np.pi])
        
    return inner, (0.0, 0.0), S, outer_rot * 180.0 / np.pi
