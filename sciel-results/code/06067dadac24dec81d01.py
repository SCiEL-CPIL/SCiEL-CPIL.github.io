import numpy as np
import time
import sys

N = 12

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # Keep 10s margin
    
    # Initial configuration: 12 hexagons in a valid 3x4 honeycomb grid
    # Flat-top hexagons
    sqrt3 = np.sqrt(3)
    
    # Generate 3x4 grid
    # Cols 0..3, Rows 0..2
    # x = 1.5 * col
    # y = sqrt(3) * (row + 0.5 * (col % 2))
    raw_pos = []
    for col in range(4):
        for row in range(3):
            x = 1.5 * col
            y = sqrt3 * (row + 0.5 * (col % 2))
            raw_pos.append([x, y])
    
    # Center the cluster
    xs = [p[0] for p in raw_pos]
    ys = [p[1] for p in raw_pos]
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    init_pos = [[p[0] - cx, p[1] - cy] for p in raw_pos]
    
    x0 = np.array([p[0] for p in init_pos])
    y0 = np.array([p[1] for p in init_pos])
    v0 = np.concatenate([x0, y0])
    
    # Normals for flat-top hexagon
    # n0 = (1, 0)
    # n1 = (0.5, sqrt(3)/2)
    # n2 = (-0.5, sqrt(3)/2)
    n0 = np.array([1.0, 0.0])
    n1 = np.array([0.5, sqrt3/2])
    n2 = np.array([-0.5, sqrt3/2])
    
    def get_vertices(x, y):
        # Returns array of shape (N, 6, 2)
        # Relative vertices for flat-top unit hex
        rel_x = np.array([1, 0.5, -0.5, -1, -0.5, 0.5])
        rel_y = np.array([0, sqrt3/2, sqrt3/2, 0, -sqrt3/2, -sqrt3/2])
        
        vx = x[:, np.newaxis] + rel_x[np.newaxis, :]
        vy = y[:, np.newaxis] + rel_y[np.newaxis, :]
        
        return vx, vy

    def overlap_penalty(x, y):
        # Calculate sum of squared penetration depths for all pairs
        # Assuming same orientation (flat-top)
        # Non-overlap condition: distance between centers must be outside Minkowski diff hex
        # Minkowski diff hex has apothem sqrt(3)
        # So, if min(|d.n0|, |d.n1|, |d.n2|) < sqrt(3), they overlap
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                
                p0 = abs(dx) # dot with n0
                p1 = abs(0.5*dx + (sqrt3/2)*dy) # dot with n1
                p2 = abs(-0.5*dx + (sqrt3/2)*dy) # dot with n2
                
                min_proj = min(p0, p1, p2)
                
                if min_proj < sqrt3:
                    # They overlap
                    pen += (sqrt3 - min_proj)**2
        return pen

    def objective(params):
        if time.time() - start_time > timeout:
            return 1e9 # Penalize timeout
            
        x = params[:N]
        y = params[N:]
        
        # 1. Calculate Outer Hex Size
        vx, vy = get_vertices(x, y)
        vx_flat = vx.flatten()
        vy_flat = vy.flatten()
        
        c1 = np.abs(vx_flat)
        c2 = np.abs(0.5 * vx_flat + (sqrt3/2) * vy_flat)
        c3 = np.abs(-0.5 * vx_flat + (sqrt3/2) * vy_flat)
        
        A = np.maximum.reduce([c1, c2, c3]).max()
        S = A * 2.0 / sqrt3
        
        # 2. Calculate Overlap Penalty
        pen = overlap_penalty(x, y)
        
        # Weighted sum
        # S is around 4-5. Penalty should be large if overlapping.
        return S + 10.0 * pen

    # Optimize
    try:
        from scipy.optimize import minimize
        # Use Powell as it is derivative-free and robust
        res = minimize(objective, v0, method='Powell', options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7})
        
        if res.success or res.fun < 1e9:
            best_params = res.x
        else:
            best_params = v0
            
        # Try Nelder-Mead if time permits
        if time.time() - start_time < timeout - 20:
             res_nm = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
             if res_nm.fun < res.fun:
                 best_params = res_nm.x

    except Exception:
        best_params = v0

    x_opt = best_params[:N]
    y_opt = best_params[N:]
    
    # Calculate final A and S
    vx, vy = get_vertices(x_opt, y_opt)
    vx_flat = vx.flatten()
    vy_flat = vy.flatten()
    c1 = np.abs(vx_flat)
    c2 = np.abs(0.5 * vx_flat + (sqrt3/2) * vy_flat)
    c3 = np.abs(-0.5 * vx_flat + (sqrt3/2) * vy_flat)
    A_opt = np.maximum.reduce([c1, c2, c3]).max()
    
    S_opt = A_opt * 2.0 / sqrt3
    
    # Construct return value
    inner = []
    for i in range(N):
        inner.append([x_opt[i], y_opt[i], 0.0])
        
    return inner, (0.0, 0.0), S_opt, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
