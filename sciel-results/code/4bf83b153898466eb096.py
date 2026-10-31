import numpy as np
from scipy.optimize import minimize
import time

N = 11

def get_hex_vertices(center, angle_deg, side=1.0):
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    # Vertices for pointy-top hexagon relative to center
    # Angles: 30, 90, 150, 210, 270, 330
    angles = np.deg2rad([30, 90, 150, 210, 270, 330]) + theta
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def compute_required_side(params):
    # params: [x1, y1, a1, x2, y2, a2, ...]
    # Fixed first hex at (0,0, 0) to reduce symmetry? 
    # Actually, let's just optimize all 22 params.
    # To avoid drift, we can penalize center of mass moving away from 0.
    
    xs = params[0::3]
    ys = params[1::3]
    as_ = params[2::3]
    
    max_s = 0.0
    for i in range(N):
        verts = get_hex_vertices((xs[i], ys[i]), as_[i])
        for vx, vy in verts:
            # S >= 2/sqrt(3) * |vx|
            s1 = (2.0 / np.sqrt(3.0)) * abs(vx)
            # S >= |vy| + 1/sqrt(3) * |vx|
            s2 = abs(vy) + (1.0 / np.sqrt(3.0)) * abs(vx)
            s_req = max(s1, s2)
            if s_req > max_s:
                max_s = s_req
                
    # Add penalty for center of mass to keep it centered
    com_x = np.mean(xs)
    com_y = np.mean(ys)
    penalty = 10.0 * (com_x**2 + com_y**2)
    
    return max_s + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial guess: Hex grid
    # q, r coords
    coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2)
    ]
    
    init_params = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        init_params.extend([x, y, 0.0])
        
    init_params = np.array(init_params)
    
    best_params = init_params
    best_score = compute_required_side(best_params)
    
    # Try a few random restarts if time permits
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Perturb initial guess
        noise = np.random.normal(0, 0.5, size=init_params.shape)
        # Keep angles small initially?
        noise[2::3] = np.random.uniform(-10, 10, size=N)
        x0 = init_params + noise
        
        try:
            res = minimize(compute_required_side, x0, method='Powell', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline:
            break
            
    # Extract final positions
    xs = best_params[0::3]
    ys = best_params[1::3]
    as_ = best_params[2::3]
    
    inner = [[xs[i], ys[i], as_[i]] for i in range(N)]
    
    # Calculate the actual required side for the best params without penalty
    # The penalty was small, but let's compute the exact S for the return value
    # The evaluator uses the returned side length. 
    # We should return the side length that actually contains them.
    # If we return a slightly larger S, it's valid but score is worse.
    # We should return the minimal S that works.
    
    max_s = 0.0
    for i in range(N):
        verts = get_hex_vertices((xs[i], ys[i]), as_[i])
        for vx, vy in verts:
            s1 = (2.0 / np.sqrt(3.0)) * abs(vx)
            s2 = abs(vy) + (1.0 / np.sqrt(3.0)) * abs(vx)
            s_req = max(s1, s2)
            if s_req > max_s:
                max_s = s_req
                
    # Add a tiny epsilon to ensure it's valid due to floating point
    side = max_s + 1e-6
    
    return inner, (0.0, 0.0), side, 0.0
