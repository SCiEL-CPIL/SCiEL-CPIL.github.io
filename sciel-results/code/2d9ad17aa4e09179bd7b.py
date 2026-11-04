import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 142.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            aj = axes_list[j]
            
            # Quick distance check
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            if dist_sq < 4.0 - 1e-9:
                return False
            
            for axis in ai:
                ax, ay = axis
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    break
            else:
                for axis in aj:
                    ax, ay = axis
                    minA = min(vx * ax + vy * ay for vx, vy in vi)
                    maxA = max(vx * ax + vy * ay for vx, vy in vi)
                    minB = min(vx * ax + vy * ay for vx, vy in vj)
                    maxB = max(vx * ax + vy * ay for vx, vy in vj)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        break
                else:
                    return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _quick_score(cx_arr, cy_arr, angle_arr):
    apothem_unit = math.sqrt(3) / 2.0
    normals = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
    max_need = 0.0
    for i in range(N):
        a = math.radians(angle_arr[i])
        ca, sa = math.cos(a), math.sin(a)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals:
                val = (nx * vx + ny * vy) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _sdf_hex_point(px, py, cx, cy, angle_deg, side=1.0):
    """Signed distance from point (px,py) to hexagon centered at (cx,cy) with given rotation.
    Negative if inside."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Transform to hexagon local frame
    dx = px - cx
    dy = py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    # For a regular hexagon with side length s, inradius = s*sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes
    inradius = side * math.sqrt(3) / 2.0
    circumradius = side
    
    # Distance to each of the 6 edges
    max_dist = -float('inf')
    for k in range(6):
        theta = math.radians(60 * k)
        # Normal pointing outward from center for edge k
        nx = math.cos(theta)
        ny = math.sin(theta)
        # Signed distance from point to this edge line (positive outside)
        d = lx * nx + ly * ny - inradius
        if d > max_dist:
            max_dist = d
    
    # If max_dist < 0, point is inside all half-planes -> inside hexagon
    # The SDF is approximately max_dist for points near the boundary
    # For a proper SDF we'd need more, but this gives a smooth-ish penalty
    return max_dist

def _sdf_hex_hex(cx1, cy1, a1, cx2, cy2, a2, side=1.0):
    """Approximate minimum separation between two hexagons using SDF.
    Returns a positive number if separated, negative if overlapping."""
    # Sample points on each hexagon's boundary
    # For efficiency, sample 12 points per hexagon
    verts1 = _hex_vertices(cx1, cy1, a1, side)
    verts2 = _hex_vertices(cx2, cy2, a2, side)
    
    min_sep = float('inf')
    for vx1, vy1 in verts1:
        d = _sdf_hex_point(vx1, vy1, cx2, cy2, a2, side)
        if d < min_sep:
            min_sep = d
    for vx2, vy2 in verts2:
        d = _sdf_hex_point(vx2, vy2, cx1, cy1, a1, side)
        if d < min_sep:
            min_sep = d
    
    # min_sep is negative if any vertex is inside the other hexagon
    # We want a penalty for overlap, so return -min_sep (positive when overlapping)
    # But we also want to penalize close approaches
    return -min_sep  # positive means overlap or touching

def _smooth_objective(params, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Objective function: maximize outer hexagon side needed + penalty for overlaps.
    We minimize this."""
    cx_arr = params[:N]
    cy_arr = params[N:2*N]
    angle_arr = params[2*N:3*N]
    
    # Compute required outer side
    a = math.radians(outer_angle)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx_c, cy_c = outer_center
    max_need = 0.0
    for i in range(N):
        ai = math.radians(angle_arr[i])
        ca, sa = math.cos(ai), math.sin(ai)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals:
                val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                if val > max_need:
                    max_need = val
    
    # Overlap penalty
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq > 16.0:  # quick skip
                continue
            sep = _sdf_hex_hex(cx_arr[i], cy_arr[i], angle_arr[i], 
                               cx_arr[j], cy_arr[j], angle_arr[j])
            if sep > 0:
                overlap_penalty += sep * sep
    
    return max_need + 10.0 * overlap_penalty

def _is_valid_sat(cx_arr, cy_arr, angle_arr):
    """SAT-based validity check, same as _is_valid_fast but kept for clarity."""
    return _is_valid_fast(cx_arr, cy_arr, angle_arr)

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_flower_initial():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        r2 = r1 * 1.732
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
        return pts[:12]
    
    def make_honeycomb_initial():
        """Honeycomb cluster: 1 center + 6 ring + 5 outer = 12"""
        pts = [[0.0, 0.0, 0.0]]
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i + 30)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        # 5 more hexagons in a second ring
        r2 = 2 * r1
        for i in range(5):
            angle = math.radians(60 * i + 60)
            x = r2 * math.cos(angle)
            y = r2 * math.sin(angle)
            pts.append([x, y, 0.0])
        return pts
    
    def make_lattice_initial():
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
        return pts
    
    def make_ring_initial():
        pts = [[0.0, 0.0, 0.0]]
        r = 2 * math.sqrt(3)
        for i in range(11):
            angle = math.radians(360 * i / 11 + 15)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            pts.append([x, y, 0.0])
        return pts
    
    def make_random_initial():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    initializers = [
        make_flower_initial,
        make_honeycomb_initial,
        make_lattice_initial,
        make_ring_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    
    # Find best valid initial configuration
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 80:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _quick_score(cx_arr, cy_arr, angle_arr)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
    
    if best_cfg is None:
        cfg = make_flower_initial()
        scale = 1.2
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side = _quick_score(cx_arr, cy_arr, angle_arr)

    # --- Phase 1: Differential Evolution ---
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in best_cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    
    pop_size = 30
    population = []
    population.append([cx_arr.copy(), cy_arr.copy(), angle_arr.copy(), best_side])
    
    for _ in range(pop_size - 1):
        p = best_cfg[:]
        for i in range(N):
            p[i][0] += random.gauss(0, 0.5)
            p[i][1] += random.gauss(0, 0.5)
            p[i][2] += random.gauss(0, 10.0)
        
        pcx = np.array([q[0] for q in p])
        pcy = np.array([q[1] for q in p])
        pang = np.array([q[2] for q in p])
        
        if _is_valid_fast(pcx, pcy, pang):
            score = _quick_score(pcx, pcy, pang)
            population.append([pcx, pcy, pang, score])
        else:
            population.append([pcx, pcy, pang, float('inf')])
    
    F = 0.8
    CR = 0.9
    de_generations = 150
    
    for gen in range(de_generations):
        if time.time() - t0 > TIME_LIMIT - 40:
            break
        
        for i in range(pop_size):
            if time.time() - t0 > TIME_LIMIT - 40:
                break
            
            a, b, c = random.sample(range(pop_size), 3)
            
            trial_cx = population[a][0].copy()
            trial_cy = population[a][1].copy()
            trial_ang = population[a][2].copy()
            
            for j in range(N):
                trial_cx[j] = population[a][0][j] + F * (population[b][0][j] - population[c][0][j])
                trial_cy[j] = population[a][1][j] + F * (population[b][1][j] - population[c][1][j])
                trial_ang[j] = population[a][2][j] + F * (population[b][2][j] - population[c][2][j])
            
            j_rand = random.randrange(N)
            for j in range(N):
                if random.random() < CR or j == j_rand:
                    cx_arr[j] = trial_cx[j]
                    cy_arr[j] = trial_cy[j]
                    angle_arr[j] = trial_ang[j]
            
            if _is_valid_fast(cx_arr, cy_arr, angle_arr):
                new_score = _quick_score(cx_arr, cy_arr, angle_arr)
                if new_score < population[i][3]:
                    population[i][0] = cx_arr.copy()
                    population[i][1] = cy_arr.copy()
                    population[i][2] = angle_arr.copy()
                    population[i][3] = new_score
                    
                    if new_score < best_side:
                        best_side = new_score
                        best_cfg = [[cx_arr[k], cy_arr[k], angle_arr[k]] for k in range(N)]
                else:
                    cx_arr[:] = population[i][0]
                    cy_arr[:] = population[i][1]
                    angle_arr[:] = population[i][2]
            else:
                cx_arr[:] = population[i][0]
                cy_arr[:] = population[i][1]
                angle_arr[:] = population[i][2]
    
    # --- Phase 2: Local Search with Simulated Annealing ---
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in best_cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    
    cur_score = best_side
    T = 0.15
    T_end = 0.001
    alpha = 0.997
    
    for step in range(6000):
        if time.time() - t0 > TIME_LIMIT - 20:
            break
        
        idx = random.randrange(N)
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        
        cx_arr[idx] += random.gauss(0, 0.05)
        cy_arr[idx] += random.gauss(0, 0.05)
        angle_arr[idx] += random.gauss(0, 1.0)
        
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _quick_score(cx_arr, cy_arr, angle_arr)
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
        else:
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
        
        T = max(T_end, T * alpha)
    
    # --- Phase 3: Gradient-based refinement using scipy ---
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = (0.0, 0.0)
    best_final_angle = 0.0
    
    # Try to refine with L-BFGS-B on the smooth objective
    # First, fix the outer hexagon at center (0,0) with angle 0 for simplicity
    # Then optimize positions
    
    if time.time() - t0 < TIME_LIMIT - 10:
        # Build parameter vector
        params0 = np.concatenate([cx_best, cy_best, angle_best])
        
        # Define objective with outer hex fixed
        def obj_smooth(params):
            if time.time() - t0 > TIME_LIMIT - 8:
                return float('inf')
            return _smooth_objective(params, (0.0, 0.0), 0.0)
        
        try:
            result = minimize(obj_smooth, params0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-8})
            if result.fun < best_side:
                new_params = result.x
                new_cx = new_params[:N]
                new_cy = new_params[N:2*N]
                new_ang = new_params[2*N:3*N]
                
                if _is_valid_fast(new_cx, new_cy, new_ang):
                    new_score = _quick_score(new_cx, new_cy, new_ang)
                    if new_score < best_final_side:
                        best_final_side = new_score
                        best_final_cfg = [[new_cx[i], new_cy[i], new_ang[i]] for i in range(N)]
        except Exception:
            pass
    
    # --- Phase 4: Optimize outer hexagon center and angle ---
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    verts_best = [_hex_vertices(cx_final[i], cy_final[i], angle_final[i]) for i in range(N)]
    
    for outer_angle in range(0, 30, 2):
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals_out = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals_out.append((math.cos(na), math.sin(na)))
        
        cx_c, cy_c = 0.0, 0.0
        
        for _iter in range(20):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
                    for (nx, ny) in normals_out:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            for dx in [-0.1, 0.0, 0.1]:
                for dy in [-0.1, 0.0, 0.1]:
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_best[i]:
                            for (nx, ny) in normals_out:
                                v = (nx * (vx - ncx) + ny * (vy - ncy)) / apothem_unit
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - cx_c) < 1e-5 and abs(best_cy - cy_c) < 1e-5:
                break
            cx_c, cy_c = best_cx, best_cy
        
        if max_proj < best_final_side:
            best_final_side = max_proj
            best_final_center = (cx_c, cy_c)
            best_final_angle = float(outer_angle)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        cfg = make_flower_initial()
        scale = 1.2
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_final_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
        best_final_center = (0.0, 0.0)
        best_final_angle = 0.0
    else:
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, best_final_center, best_final_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_final_cfg]
    return inner, best_final_center, best_final_side, best_final_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
