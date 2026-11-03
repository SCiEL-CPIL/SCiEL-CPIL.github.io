import math
import numpy as np
from scipy.optimize import differential_evolution

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    rel = [
        (side, 0.0),
        (side * 0.5, side * math.sqrt(3) / 2),
        (-side * 0.5, side * math.sqrt(3) / 2),
        (-side, 0.0),
        (-side * 0.5, -side * math.sqrt(3) / 2),
        (side * 0.5, -side * math.sqrt(3) / 2),
    ]
    verts = []
    for vx, vy in rel:
        rx = cx + vx * ca - vy * sa
        ry = cy + vx * sa + vy * ca
        verts.append((rx, ry))
    return np.array(verts)

def _min_outer_side_for_vertices(verts, angle_rad):
    s3 = math.sqrt(3)
    normal_angles = [math.radians(30 + k * 60) for k in range(6)]
    normals = np.array([(math.cos(a), math.sin(a)) for a in normal_angles])
    
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated_verts = verts @ R.T
    dots = rotated_verts @ normals.T
    abs_dots = np.abs(dots)
    
    max_proj = np.max(abs_dots)
    S = 2.0 * max_proj / s3
    return S

def _is_strictly_inside(p, poly):
    n = len(poly)
    signs = []
    for i in range(n):
        a = poly[i]
        b = poly[(i+1)%n]
        cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        if abs(cross) < 1e-9:
            return False
        signs.append(np.sign(cross))
    if all(s == signs[0] for s in signs):
        return True
    return False

def _check_validity(hexes):
    N = len(hexes)
    for i in range(N):
        for j in range(i+1, N):
            v1 = hexes[i]
            v2 = hexes[j]
            
            # SAT Check
            separated = False
            # Get normals for v1
            for i_edge in range(6):
                p1 = v1[i_edge]
                p2 = v1[(i_edge+1)%6]
                edge = p2 - p1
                norm = np.array([edge[1], -edge[0]])
                norm_len = np.linalg.norm(norm)
                if norm_len > 1e-9:
                    norm = norm / norm_len
                    proj1 = np.dot(v1, norm)
                    proj2 = np.dot(v2, norm)
                    if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                        separated = True
                        break
            if separated:
                continue
                
            # Get normals for v2
            for i_edge in range(6):
                p1 = v2[i_edge]
                p2 = v2[(i_edge+1)%6]
                edge = p2 - p1
                norm = np.array([edge[1], -edge[0]])
                norm_len = np.linalg.norm(norm)
                if norm_len > 1e-9:
                    norm = norm / norm_len
                    proj1 = np.dot(v1, norm)
                    proj2 = np.dot(v2, norm)
                    if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                        separated = True
                        break
            if separated:
                continue
            
            # Not separated: check for strict intersection
            intersects = False
            for p in v1:
                if _is_strictly_inside(p, v2):
                    intersects = True
                    break
            if not intersects:
                for p in v2:
                    if _is_strictly_inside(p, v1):
                        intersects = True
                        break
                        
            if intersects:
                return False
            # Else touching, which is valid
            
    return True

def _evaluate_config(params):
    N = 11
    outer_angle = params[-1]
    
    hexes = []
    all_verts = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang_deg = params[3*i+2]
        verts = _hex_vertices(cx, cy, ang_deg, 1.0)
        hexes.append(verts)
        all_verts.append(verts)
        
    all_verts = np.vstack(all_verts)
    
    if not _check_validity(hexes):
        return 1e10
        
    side = _min_outer_side_for_vertices(all_verts, outer_angle)
    return side

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 11
    s3 = math.sqrt(3)
    
    centers = [(0.0, 0.0)]
    for k in range(6):
        ang = math.radians(60 * k)
        centers.append((s3 * math.cos(ang), s3 * math.sin(ang)))
    for k in range(4):
        ang = math.radians(90 * k)
        centers.append((2*s3 * math.cos(ang), 2*s3 * math.sin(ang)))
        
    params_init = np.zeros(3*N + 1)
    for i, (cx, cy) in enumerate(centers):
        params_init[3*i] = cx
        params_init[3*i+1] = cy
        params_init[3*i+2] = 0.0
    params_init[-1] = 0.0
    
    bounds = [(-5, 5), (-5, 5), (0, 60)] * N + [(0, math.pi/3)]
    
    best_side = float('inf')
    best_params = params_init.copy()
    
    starts = [params_init]
    for _ in range(2):
        p = np.zeros(3*N + 1)
        for i in range(N):
            p[3*i] = np.random.uniform(-3, 3)
            p[3*i+1] = np.random.uniform(-3, 3)
            p[3*i+2] = np.random.uniform(0, 60)
        p[-1] = np.random.uniform(0, math.pi/3)
        starts.append(p)
        
    for start in starts:
        if time.time() > deadline:
            break
            
        try:
            result = differential_evolution(
                _evaluate_config,
                bounds=bounds,
                x0=start,
                seed=42,
                maxiter=50,
                popsize=8,
                tol=1e-3,
                init='sobol'
            )
            
            if result.fun < 1e9 and result.fun < best_side:
                best_side = result.fun
                best_params = result.x
        except Exception:
            pass
            
    if best_side >= 1e9:
        val = _evaluate_config(params_init)
        if val < 1e9:
            best_side = val
            best_params = params_init
        else:
            p_line = np.zeros(3*N + 1)
            for i in range(N):
                p_line[3*i] = i * s3
                p_line[3*i+1] = 0.0
                p_line[3*i+2] = 0.0
            p_line[-1] = 0.0
            val = _evaluate_config(p_line)
            if val < 1e9:
                best_side = val
                best_params = p_line
            else:
                # Fallback
                best_side = _evaluate_config(params_init)
                best_params = params_init
                
    inner_list = []
    for i in range(N):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        ang_deg = best_params[3*i+2]
        inner_list.append((cx, cy, ang_deg))
        
    outer_angle = best_params[-1]
    center = (0.0, 0.0)
    side = best_side + 1e-4
    
    return (inner_list, center, side, outer_angle)
