import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580  # 20s margin
    
    D = np.sqrt(3)  # Min distance between centers for angle 30
    eps = 1e-4      # Safety gap
    
    # Initial configuration: Hexagonal cluster
    # Axial coords for 12 hexagons
    # 1 center, 6 ring 1, 5 ring 2
    # We want a compact shape.
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Add 5 from ring 2:
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1) ... wait, (-1,-1) is ring 1? No, ring 1 is dist 1.
    # Axial distance: |q| + |r| + |q+r| / 2 ? No.
    # Let's just use a grid and pick 12 closest to origin.
    
    coords = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            # Axial to Cartesian (flat-top, angle 30)
            # x = sqrt(3) * (q + r/2)
            # y = 1.5 * r
            x = np.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            coords.append((x, y))
            
    # Sort by distance to origin
    coords.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take first 12
    init_positions = np.array(coords[:12])
    
    # Center the cluster
    center = init_positions.mean(axis=0)
    init_positions = init_positions - center
    
    # Initial s
    # Max distance from origin to any vertex of any hexagon
    # Vertex offset for side 1, angle 30:
    # Apothem = sqrt(3)/2
    # Max radial extent is 1 (vertex)
    # So max radius of cluster + 1
    max_r = np.max(np.linalg.norm(init_positions, axis=1))
    s_init = max_r + 1.0 + eps
    
    # Variables: x1, y1, ..., x12, y12, s
    x0 = np.concatenate([init_positions.flatten(), [s_init]])
    
    def objective(vars):
        s = vars[-1]
        # s must be positive
        if s <= 0:
            return 1e10
        return s

    def constraints(vars):
        cons = []
        s = vars[-1]
        pos = vars[:-1].reshape(12, 2)
        
        # 1. Distance constraints
        for i in range(12):
            for j in range(i+1, 12):
                diff = pos[i] - pos[j]
                dist_sq = np.dot(diff, diff)
                min_dist = D + eps
                # dist >= min_dist  =>  dist_sq - min_dist^2 >= 0
                cons.append({'type': 'ineq', 'fun': lambda d=dist_sq: d - min_dist**2})
                
        # 2. Containment constraints
        # Outer hexagon center (0,0), angle 0, side s.
        # Normals: 0, 60, 120, 180, 240, 300
        # Apothem A = s * sqrt(3) / 2
        A = s * np.sqrt(3) / 2.0
        
        # Inner hexagon vertices relative to center
        # Angle 30 deg
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        offsets = np.column_stack([np.cos(angles), np.sin(angles)])
        
        for i in range(12):
            cx, cy = pos[i]
            verts = cx + offsets[:, 0]
            vertys = cy + offsets[:, 1]
            
            for k in range(6):
                vx = verts[k]
                vy = vertys[k]
                for n in range(6):
                    ang = np.deg2rad(n * 60)
                    nx = np.cos(ang)
                    ny = np.sin(ang)
                    # nx * vx + ny * vy <= A
                    # nx * vx + ny * vy - A <= 0
                    # A = s * sqrt(3) / 2
                    # nx * vx + ny * vy - s * sqrt(3) / 2 <= 0
                    cons.append({'type': 'ineq', 'fun': lambda v=vx, w=vy, nx=nx, ny=ny: A - (nx * v + ny * w)})
                    
        return cons

    # Optimize
    best_result = None
    best_s = s_init
    
    # Use Nelder-Mead first for robustness, then maybe polish
    # Or SLSQP
    try:
        res = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-8})
        if res.success and res.fun < best_s:
            best_s = res.fun
            best_vars = res.x
        else:
            best_vars = x0
    except Exception:
        best_vars = x0

    # If time remains, try Nelder-Mead on the result
    if time.time() < deadline:
        try:
            res2 = minimize(objective, best_vars, method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res2.fun < best_s:
                best_s = res2.fun
                best_vars = res2.x
        except Exception:
            pass

    # Extract final positions and s
    final_pos = best_vars[:-1].reshape(12, 2)
    final_s = best_vars[-1]
    
    # Construct return arrays
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_pos[i, 0]
        inner_hex_data[i, 1] = final_pos[i, 1]
        inner_hex_data[i, 2] = 30.0  # Angle in degrees
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, final_s

# EVOLVE-BLOCK-END
