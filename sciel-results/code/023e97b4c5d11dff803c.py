import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_support(x, y, angle_deg, ux, uy):
    """Support function of a unit regular hexagon centered at (x,y) with given angle."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Vertices of unit regular hexagon (side=1, circumradius=1)
    k = np.arange(6)
    verts_x = np.cos(k * np.pi / 3)
    verts_y = np.sin(k * np.pi / 3)
    # Rotate
    rx = cos_a * verts_x - sin_a * verts_y
    ry = sin_a * verts_x + cos_a * verts_y
    # Translate
    px = rx + x
    py = ry + y
    return np.max(px * ux + py * uy)

def outer_hex_support(cx, cy, S, angle_deg, ux, uy):
    """Support function of outer hexagon with side length S."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    k = np.arange(6)
    verts_x = S * np.cos(k * np.pi / 3)
    verts_y = S * np.sin(k * np.pi / 3)
    rx = cos_a * verts_x - sin_a * verts_y
    ry = sin_a * verts_x + cos_a * verts_y
    px = rx + cx
    py = ry + cy
    return np.max(px * ux + py * uy)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Honeycomb initial arrangement
    positions = []
    angles = []
    dx = np.sqrt(3)
    dy = 1.5
    
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y])
            angles.append(0.0)
    
    positions = np.array(positions)
    positions -= positions.mean(axis=0)
    
    cx, cy = 0.0, 0.0
    S_init = 4.5
    outer_angle = 0.0
    
    def unpack(v):
        pos = v[:36].reshape(12, 3)
        return pos, v[36], v[37], v[38], v[39]
    
    def objective(v):
        return v[38]
    
    def constraints(v):
        pos, cx, cy, S, outer_angle = unpack(v)
        cons = []
        
        # Outer hexagon normals
        oa = np.radians(outer_angle)
        outer_normals = []
        for k in range(6):
            theta = oa + np.pi/6 + k * np.pi / 3
            outer_normals.append((np.cos(theta), np.sin(theta)))
        
        # Containment: for each inner hex, each outer normal direction
        for i in range(12):
            xi, yi, ai = pos[i]
            for (nx, ny) in outer_normals:
                sup_i = hex_support(xi, yi, ai, nx, ny)
                sup_o = outer_hex_support(cx, cy, S, outer_angle, nx, ny)
                cons.append({'type': 'ineq', 'fun': lambda s_i=sup_i, s_o=sup_o: s_o - s_i})
        
        # Disjointness: for each pair, check 12 critical directions
        for i in range(12):
            for j in range(i+1, 12):
                xi, yi, ai = pos[i]
                xj, yj, aj = pos[j]
                
                dirs_check = []
                for k in range(6):
                    theta = np.radians(ai) + np.pi/6 + k * np.pi / 3
                    dirs_check.append((np.cos(theta), np.sin(theta)))
                for k in range(6):
                    theta = np.radians(aj) + np.pi/6 + k * np.pi / 3
                    dirs_check.append((np.cos(theta), np.sin(theta)))
                
                for (ux, uy) in dirs_check:
                    sup_i = hex_support(xi, yi, ai, ux, uy)
                    sup_j_neg = hex_support(xj, yj, aj, -ux, -uy)
                    cons.append({'type': 'ineq', 'fun': lambda s1=sup_i, s2=sup_j_neg: s1 + s2})
        
        return cons
    
    v0 = np.zeros(40)
    for i in range(12):
        v0[3*i] = positions[i, 0]
        v0[3*i+1] = positions[i, 1]
        v0[3*i+2] = 0.0
    v0[36] = 0.0
    v0[37] = 0.0
    v0[38] = S_init
    v0[39] = 0.0
    
    bounds = [(-10, 10), (-10, 10), (0, 180)] * 12 + [(-10, 10), (-10, 10), (2.0, 10.0), (0, 60)]
    
    best_S = np.inf
    best_v = v0.copy()
    
    np.random.seed(42)
    
    for restart in range(8):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            v_current = v0.copy()
        else:
            v_current = best_v + np.random.normal(0, 0.5, 40)
            v_current[38] = max(2.0, min(10.0, v_current[38]))
        
        try:
            result = minimize(objective, v_current, method='SLSQP', 
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 300, 'ftol': 1e-9})
            
            if result.fun < best_S:
                best_S = result.fun
                best_v = result.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5:
            break
    
    pos, cx, cy, S, outer_angle = unpack(best_v)
    
    inner = []
    for i in range(12):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])])
    
    return inner, (float(cx), float(cy)), float(S), float(outer_angle)

