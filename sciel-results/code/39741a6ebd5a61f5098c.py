import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _find_min_enclosing_hex_side_lp(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices using LP.
    Returns (center_x, center_y, side_length).
    """
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            # Constraint 1: -nx*cx - ny*cy - R <= -v_dot_n
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
            
            # Constraint 2: nx*cx + ny*cy - R <= v_dot_n
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            
    # R >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except:
        return None

def _get_best_angle_and_side(positions, inner_angle):
    """
    Find the best outer angle and corresponding min side length for given positions.
    """
    all_vertices = []
    for px, py in positions:
        verts = _get_hex_vertices(px, py, inner_angle)
        all_vertices.extend(verts)
        
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Coarse search
    for outer_angle_deg in np.arange(0, 30.1, 0.5):
        if time.time() > _DEADLINE - 15.0:
            break
        res = _find_min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = outer_angle_deg
                best_center = (cx, cy)
                
    # Fine search
    if best_angle > 0:
        search_range = 0.5
        step = 0.01
        for outer_angle_deg in np.arange(best_angle - search_range, best_angle + search_range + step, step):
            if outer_angle_deg < 0 or outer_angle_deg > 30:
                continue
            if time.time() > _DEADLINE - 15.0:
                break
            res = _find_min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_angle = outer_angle_deg
                    best_center = (cx, cy)
                    
    return best_side, best_angle, best_center

def _overlap_penalty(positions):
    min_dist = math.sqrt(3) - 1e-5
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = positions[2*i] - positions[2*j]
            dy = positions[2*i+1] - positions[2*j+1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    return penalty

def _smooth_objective(positions, inner_angle, fixed_outer_angle):
    all_vertices = []
    for i in range(N):
        px = positions[2*i]
        py = positions[2*i+1]
        verts = _get_hex_vertices(px, py, inner_angle)
        all_vertices.extend(verts)
        
    res = _find_min_enclosing_hex_side_lp(all_vertices, fixed_outer_angle)
    if res:
        side = res[2]
    else:
        side = 100.0
        
    pen = _overlap_penalty(positions)
    return side + 10.0 * pen

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Generate 11 points on triangular lattice
    sqrt3 = math.sqrt(3)
    scale = 1.000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
            
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    best_positions = None
    
    # Try inner angles 0 and 30
    for inner_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 20.0:
            break
            
        # 1. Find best angle for initial layout
        side, angle, center = _get_best_angle_and_side(positions, inner_angle)
        
        if side < best_side:
            best_side = side
            best_angle = angle
            best_center = center
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_outer_angle = angle
            best_inner_angle = inner_angle
            best_positions = positions[:]
            
    # 2. Optimize positions using Nelder-Mead
    if best_inner is not None:
        current_positions = np.array([coord for pos in best_positions for coord in pos])
        current_inner_angle = best_inner_angle
        current_outer_angle = best_outer_angle
        
        # Define objective
        def obj(pos_flat):
            if time.time() > _DEADLINE - 10.0:
                return float('inf')
            return _smooth_objective(pos_flat, current_inner_angle, current_outer_angle)
            
        # Nelder-Mead options
        options = {
            'maxiter': 300,
            'xatol': 1e-4,
            'fatol': 1e-4,
            'adaptive': True
        }
        
        try:
            res = minimize(obj, current_positions, method='Nelder-Mead', options=options)
            if res.success and res.fun < best_side:
                # Update best
                # Check if penalty is 0
                pen = _overlap_penalty(res.x)
                if pen < 1e-8:
                    best_side = res.fun
                    new_positions = [(res.x[2*i], res.x[2*i+1]) for i in range(N)]
                    
                    # Re-find best angle for new positions
                    new_side, new_angle, new_center = _get_best_angle_and_side(new_positions, current_inner_angle)
                    
                    if new_side < best_side:
                        best_side = new_side
                        best_outer_angle = new_angle
                        best_center = new_center
                        best_inner = [[pos[0], pos[1], current_inner_angle] for pos in new_positions]
                        
        except Exception as e:
            pass # Keep previous best

    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
