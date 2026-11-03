import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    # ... (Same as current) ...
    pass

def _compute_lse_objective(z, beta=10.0):
    # z = [cx, cy, theta, x1, y1, a1, ..., x11, y11, a11]
    cx, cy = z[0], z[1]
    theta = z[2]
    
    # Normals
    base_angle = theta + math.pi / 6 # 30 deg offset
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Calculate projections
    # We need vertices of all 11 hexagons
    # Loop over hexagons
    max_proj = -1e9
    sum_exp = 0.0
    
    for i in range(N):
        xi = z[3 + i*3]
        yi = z[4 + i*3]
        ai_deg = z[5 + i*3]
        ai = math.radians(ai_deg)
        
        # Vertices of this hex
        v_angles = ai + np.arange(6) * math.pi / 3.0
        vx = xi + np.cos(v_angles)
        vy = yi + np.sin(v_angles)
        
        # Projections onto 6 normals
        # (vx - cx) * nx + (vy - cy) * ny
        for k in range(6):
            nx, ny = normals[k]
            proj = (vx - cx) * nx + (vy - cy) * ny
            # Accumulate for LSE
            # To avoid overflow, we can do it in log space or just be careful
            # proj is likely small (< 10). exp(10*10) is huge.
            # Let's shift by max_proj first? No, we need the sum.
            # We can just compute exp(beta * proj) if beta * proj < 700.
            # With beta=10, proj<70 is safe.
            sum_exp += np.sum(np.exp(beta * proj))
            
    if sum_exp == 0:
        return 1e9
        
    lse_val = (1.0 / beta) * math.log(sum_exp)
    
    # Overlap Penalty
    penalty = 0.0
    positions = np.array([[z[3+i*3], z[4+i*3]] for i in range(N)])
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(positions[i] - positions[j])
            # Minimum distance between centers for disjoint hexagons is sqrt(3)
            # If d < sqrt(3), they might overlap.
            # Penalty if d < sqrt(3)
            if d < _SQRT3:
                penalty += (_SQRT3 - d)**2
                
    return lse_val + 10.0 * penalty

def _optimize_layout(positions, inner_angles, cx_init, cy_init, theta_init, time_limit=10.0):
    # z = [cx, cy, theta, x1, y1, a1, ...]
    z0 = [cx_init, cy_init, math.radians(theta_init)]
    for i in range(N):
        z0.extend([positions[i][0], positions[i][1], inner_angles[i]])
        
    start_time = time.time()
    
    def obj(z):
        if time.time() > start_time + time_limit:
            return 1e9
        return _compute_lse_objective(z)
        
    # Bounds
    # cx, cy: [-10, 10]
    # theta: [0, 30] degrees -> radians
    # x, y: [-10, 10]
    # angle: [-180, 180]
    bounds = [(-10, 10), (-10, 10), (0, math.pi/6)]
    for i in range(N):
        bounds.extend([(-10, 10), (-10, 10), (-math.pi, math.pi)])
        
    try:
        res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-9})
        if res.success or res.fun < 1e8:
            return res.x
    except Exception:
        pass
    return z0

def construct():
    # ... (Setup) ...
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # ... (Generate layouts) ...
    
    for layout_idx, positions in enumerate(layouts):
        # ... (Coarse search for init) ...
        
        # Optimize
        z_opt = _optimize_layout(positions, inner_angles, cx_init, cy_init, theta_init)
        
        # Extract optimized positions
        opt_positions = [(z_opt[3+i*3], z_opt[4+i*3]) for i in range(N)]
        opt_angles = [math.degrees(z_opt[5+i*3]) for i in range(N)]
        opt_cx, opt_cy = z_opt[0], z_opt[1]
        opt_theta = math.degrees(z_opt[2])
        
        # Verify with LP
        all_vertices = np.vstack([_hex_vertices(px, py, aa) for px, py, aa in zip(opt_positions, opt_angles)])
        res = _min_enclosing_hex_side(all_vertices, opt_theta)
        
        if res:
            cx, cy, side = res
            # Check disjointness
            if _check_disjoint(opt_positions, opt_angles):
                if side < best_side:
                    best_side = side
                    best_inner = [[px, py, aa] for px, py, aa in zip(opt_positions, opt_angles)]
                    best_center = (cx, cy)
                    best_outer_angle = opt_theta
                    
    # ... (Return) ...
