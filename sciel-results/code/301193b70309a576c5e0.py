import math
import time
import numpy as np
from scipy.optimize import linprog, differential_evolution, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0

def _get_hex_vertices(cx, cy, angle_deg):
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return np.array(verts)

def _get_hex_normals(angle_deg):
    angle_rad = math.radians(angle_deg)
    normals = []
    for k in range(6):
        a = angle_rad + math.pi / 6 + math.pi * k / 3
        normals.append((math.cos(a), math.sin(a)))
    return np.array(normals)

def _penetration_depth(verts1, normals1, verts2, normals2):
    min_depth = float('inf')
    all_normals = np.vstack([normals1, normals2])
    for n in all_normals:
        proj1 = verts1.dot(n)
        min1, max1 = proj1.min(), proj1.max()
        proj2 = verts2.dot(n)
        min2, max2 = proj2.min(), proj2.max()
        if max1 <= min2 or max2 <= min1:
            return 0.0
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    return min_depth

def _calc_cost(params):
    if time.time() > _DEADLINE - 2.0:
        return float('inf')
        
    hex_data = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        verts = _get_hex_vertices(x, y, a)
        normals = _get_hex_normals(a)
        hex_data.append((verts, normals))
        
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            v1, n1 = hex_data[i]
            v2, n2 = hex_data[j]
            depth = _penetration_depth(v1, n1, v2, n2)
            if depth > 1e-6:
                penalty += depth * depth
                
    if penalty > 1.0:
        return 1000.0 + penalty

    all_vertices = []
    for v, _ in hex_data:
        all_vertices.extend(v)
    all_vertices = np.array(all_vertices)
    
    best_side = float('inf')
    
    for outer_angle in [0.0, 15.0, 30.0, 45.0]:
        normals = _get_hex_normals(outer_angle)
        A = []
        b = []
        for vx, vy in all_vertices:
            for nx, ny in normals:
                v_dot_n = vx * nx + vy * ny
                A.append([-nx, -ny, -1.0])
                b.append(-v_dot_n)
                A.append([nx, ny, -1.0])
                b.append(v_dot_n)
        A.append([0.0, 0.0, -1.0])
        b.append(0.0)
        c_obj = [0.0, 0.0, 1.0]
        try:
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
            if res.success:
                R = res.x[2]
                side = R / (math.sqrt(3) / 2)
                if side < best_side:
                    best_side = side
        except:
            pass
            
    if best_side == float('inf'):
        return 1000.0 + penalty
        
    return best_side + 10.0 * penalty

def _get_outer_hex_for_inner(inner):
    all_vertices = []
    for p in inner:
        all_vertices.extend(_get_hex_vertices(p[0], p[1], p[2]))
        
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for outer_angle in [0.0, 15.0, 30.0, 45.0]:
        normals = _get_hex_normals(outer_angle)
        A = []
        b = []
        for vx, vy in all_vertices:
            for nx, ny in normals:
                v_dot_n = vx * nx + vy * ny
                A.append([-nx, -ny, -1.0])
                b.append(-v_dot_n)
                A.append([nx, ny, -1.0])
                b.append(v_dot_n)
        A.append([0.0, 0.0, -1.0])
        b.append(0.0)
        c_obj = [0.0, 0.0, 1.0]
        try:
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
            if res.success:
                side = res.x[2] / (math.sqrt(3) / 2)
                if side < best_side:
                    best_side = side
                    best_center = (res.x[0], res.x[1])
                    best_angle = outer_angle
        except:
            pass
            
    return best_center, best_side, best_angle

def _is_valid(inner):
    hex_data = []
    for p in inner:
        v = _get_hex_vertices(p[0], p[1], p[2])
        n = _get_hex_normals(p[2])
        hex_data.append((v, n))
        
    for i in range(N):
        for j in range(i+1, N):
            depth = _penetration_depth(hex_data[i][0], hex_data[i][1], hex_data[j][0], hex_data[j][1])
            if depth > 1e-5:
                return False
    return True

def _fallback_solution():
    sqrt3 = math.sqrt(3)
    positions = []
    for i in [-1, 0, 1]:
        positions.append((i * sqrt3, 1.5))
    for i in [-1.5, -0.5, 0.5, 1.5]:
        positions.append((i * sqrt3, 0.0))
    for i in [-1, 0, 1]:
        positions.append((i * sqrt3, -1.5))
        
    cx = sum(p[0] for p in positions) / N
    cy = sum(p[1] for p in positions) / N
    positions = [(p[0]-cx, p[1]-cy) for p in positions]
    inner = [[p[0], p[1], 0.0] for p in positions]
    
    center, side, angle = _get_outer_hex_for_inner(inner)
    if side == float('inf'):
        return inner, (0.0, 0.0), 10.0, 0.0
    return inner, center, side, angle

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        return _fallback_solution()

    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 60.0))
        
    best_params = None
    best_cost = float('inf')
    
    # Run DE
    try:
        res = differential_evolution(
            _calc_cost, 
            bounds, 
            seed=42, 
            maxiter=30, 
            popsize=10, 
            tol=1e-4,
            polish=False
        )
        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
    except Exception:
        pass
        
    if time.time() > _DEADLINE - 15.0:
        if best_params is not None:
            inner = []
            for i in range(N):
                x = best_params[3*i]
                y = best_params[3*i+1]
                a = best_params[3*i+2] % 60.0
                inner.append([x, y, a])
            if _is_valid(inner):
                center, side, angle = _get_outer_hex_for_inner(inner)
                if side != float('inf'):
                    return inner, center, side, angle
        return _fallback_solution()

    # Refine
    if best_params is not None:
        try:
            res_refine = minimize(
                _calc_cost, 
                best_params, 
                method='Powell', 
                options={'maxiter': 200, 'xatol': 1e-6}
            )
            if res_refine.fun < best_cost:
                best_cost = res_refine.fun
                best_params = res_refine.x
        except Exception:
            pass
            
    if best_params is None:
        return _fallback_solution()
        
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2] % 60.0
        inner.append([x, y, a])
        
    if not _is_valid(inner):
        return _fallback_solution()
        
    center, side, angle = _get_outer_hex_for_inner(inner)
    if side == float('inf'):
        return _fallback_solution()
        
    return inner, center, side, angle
