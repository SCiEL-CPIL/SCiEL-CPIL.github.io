import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s

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

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            vj = verts_list[j]
            aj = axes_list[j]
            
            dx = xi - xj
            dy = yi - yj
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

def _smooth_collision_penalty(cx_arr, cy_arr, angle_arr, eps=0.05):
    """Compute smooth collision penalty using distance between hexagon boundaries."""
    n = len(cx_arr)
    penalty = 0.0
    for i in range(n):
        vi = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for j in range(i + 1, n):
            vj = _hex_vertices(cx_arr[j], cy_arr[j], angle_arr[j])
            # Compute minimum distance between the two polygons
            min_dist = float('inf')
            for p1 in vi:
                for p2 in vj:
                    d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    if d < min_dist:
                        min_dist = d
            # Also check edge-to-edge distances
            for e1_i in range(6):
                p1a = vi[e1_i]
                p1b = vi[(e1_i + 1) % 6]
                for e2_j in range(6):
                    p2a = vj[e2_j]
                    p2b = vj[(e2_j + 1) % 6]
                    # Distance from point p1a to segment p2a-p2b
                    def point_segment_dist(p, a, b):
                        dx = b[0] - a[0]
                        dy = b[1] - a[1]
                        if dx == 0 and dy == 0:
                            return math.hypot(p[0] - a[0], p[1] - a[1])
                        t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx*dx + dy*dy)))
                        px = a[0] + t * dx
                        py = a[1] + t * dy
                        return math.hypot(p[0] - px, p[1] - py)
                    
                    d1 = point_segment_dist(p1a, p2a, p2b)
                    d2 = point_segment_dist(p1b, p2a, p2b)
                    d3 = point_segment_dist(p2a, p1a, p1b)
                    d4 = point_segment_dist(p2b, p1a, p1b)
                    edge_dist = min(d1, d2, d3, d4)
                    if edge_dist < min_dist:
                        min_dist = edge_dist
            
            if min_dist < eps:
                # Smooth penalty: quadratic in (eps - min_dist)
                penalty += (eps - min_dist) ** 2
    
    return penalty

def _surrogate_objective(params, cx_arr, cy_arr, angle_arr):
    """
    Smooth surrogate: max projection onto outer hexagon normals + collision penalty.
    params: [cx_offset, cy_offset] for outer hexagon center
    """
    cx_off, cy_off = params
    outer_cx, outer_cy = 0.0 + cx_off, 0.0 + cy_off
    
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    max_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - outer_cx) + ny * (vy - outer_cy)) / apothem_unit
                if val > max_proj:
                    max_proj = val
    
    collision_pen = _smooth_collision_penalty(cx_arr, cy_arr, angle_arr)
    
    return max_proj + 10.0 * collision_pen

def _gradient_refine(cx_arr, cy_arr, angle_arr, time_budget):
    """
    Use L-BFGS-B to refine positions by minimizing a smooth surrogate.
    Returns improved (cx_arr, cy_arr, angle_arr) if valid and better.
    """
    t_start = time.time()
    
    # Flatten parameters: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cx_off, cy_off]
    params = np.concatenate([
        np.array([cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)),
        np.array([0.0, 0.0])
    ])
    
    def objective(p):
        if time.time() - t_start > time_budget:
            return 1e10
        cx = p[0::3][:N]
        cy = p[1::3][:N]
        ang = p[2::3][:N]
        cx_off = p[-2]
        cy_off = p[-1]
        
        # Quick validity check using center distances
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                if dx*dx + dy*dy < 3.5:  # Slightly less than 4.0 for margin
                    return 1e10 + (4.0 - dx*dx - dy*dy) * 100
        
        return _surrogate_objective([cx_off, cy_off], cx, cy, ang)
    
    try:
        result = minimize(
            objective,
            params,
            method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 200}
        )
    except Exception:
        return None
    
    if time.time() - t_start > time_budget:
        return None
    
    new_params = result.x
    new_cx = new_params[0::3][:N]
    new_cy = new_params[1::3][:N]
    new_ang = new_params[2::3][:N]
    
    # Verify validity
    if not _is_valid_fast(new_cx, new_cy, new_ang):
        return None
    
    # Compute actual side length
    side = _outer_hex_side(new_cx, new_cy, new_ang, (0.0, 0.0), 0.0)
    
    return new_cx, new_cy, new_ang, side

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
    
    def make_random_initial():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
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

    initializers = [
        make_flower_initial,
        make_lattice_initial,
        make_ring_initial,
        make_random_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
                best_center = (0.0, 0.0)
                best_angle = 0.0
    
    if best_cfg is None:
        cfg = make_lattice_initial()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # --- Phase 1: Simulated Annealing ---
    cfg = [row[:] for row in best_cfg]
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    cur_score = best_side
    T = 0.3
    T_end = 0.001
    alpha = 0.995
    
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    def _quick_score(cx_arr, cy_arr, angle_arr):
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

    # Main SA loop with periodic restarts
    sa_restarts = 3
    steps_per_restart = 5000
    
    for restart in range(sa_restarts):
        if time.time() - t0 > TIME_LIMIT - 50:
            break
        
        if restart > 0:
            cx_arr[:] = [p[0] for p in best_cfg]
            cy_arr[:] = [p[1] for p in best_cfg]
            angle_arr[:] = [p[2] for p in best_cfg]
            cur_score = best_side
            T = 0.2
        
        for step in range(steps_per_restart):
            if time.time() - t0 > TIME_LIMIT - 50:
                break
            
            r = random.random()
            idx = random.randrange(N)
            old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
            idx2 = -1
            old_x2, old_y2, old_a2 = 0, 0, 0
            
            if r < 0.65:
                cx_arr[idx] += random.gauss(0, 0.10)
                cy_arr[idx] += random.gauss(0, 0.10)
                angle_arr[idx] += random.gauss(0, 2.0)
            elif r < 0.80:
                angle_arr[idx] += random.gauss(0, 5.0)
            elif r < 0.90:
                idx2 = random.randrange(N)
                if idx2 == idx:
                    idx2 = (idx + 1) % N
                old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
                cx_arr[idx], cx_arr[idx2] = cx_arr[idx2], cx_arr[idx]
                cy_arr[idx], cy_arr[idx2] = cy_arr[idx2], cy_arr[idx]
                angle_arr[idx], angle_arr[idx2] = angle_arr[idx2], angle_arr[idx]
            else:
                idx2 = random.randrange(N)
                if idx2 == idx:
                    idx2 = (idx + 1) % N
                old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
                cx_arr[idx] += random.gauss(0, 0.06)
                cy_arr[idx] += random.gauss(0, 0.06)
                cx_arr[idx2] += random.gauss(0, 0.06)
                cy_arr[idx2] += random.gauss(0, 0.06)
                angle_arr[idx] += random.gauss(0, 1.0)
                angle_arr[idx2] += random.gauss(0, 1.0)
            
            valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
            
            if valid:
                new_score = _quick_score(cx_arr, cy_arr, angle_arr)
                if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                    cur_score = new_score
                    if new_score < best_side:
                        best_side = new_score
                        best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                        best_center = (0.0, 0.0)
                        best_angle = 0.0
                else:
                    cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                    if idx2 != -1:
                        cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if idx2 != -1:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
            
            T = max(T_end, T * alpha)
    
    # --- Phase 2: Gradient-based refinement using L-BFGS-B ---
    if time.time() - t0 < TIME_LIMIT - 20:
        cx_best = np.array([p[0] for p in best_cfg])
        cy_best = np.array([p[1] for p in best_cfg])
        angle_best = np.array([p[2] for p in best_cfg])
        
        remaining = TIME_LIMIT - 20 - (time.time() - t0)
        if remaining > 5:
            refined = _gradient_refine(cx_best, cy_best, angle_best, remaining)
            if refined is not None:
                new_cx, new_cy, new_ang, new_side = refined
                if new_side < best_side:
                    best_side = new_side
                    best_cfg = [[new_cx[i], new_cy[i], new_ang[i]] for i in range(N)]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
    
    # --- Phase 3: Optimize outer hexagon center and angle for best inner config ---
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle
    
    verts_best = [_hex_vertices(cx_best[i], cy_best[i], angle_best[i]) for i in range(N)]
    
    for outer_angle in range(0, 30, 1):
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals_out = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals_out.append((math.cos(na), math.sin(na)))
        
        cx_c, cy_c = best_final_center[0], best_final_center[1]
        
        for _iter in range(30):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
                    for (nx, ny) in normals_out:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            for dx in [-0.15, -0.08, 0, 0.08, 0.15]:
                for dy in [-0.15, -0.08, 0, 0.08, 0.15]:
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
    
    # --- Phase 4: Final fine-tuning with SA on the best config ---
    if time.time() - t0 < TIME_LIMIT - 10:
        cx_arr2 = np.array([p[0] for p in best_final_cfg])
        cy_arr2 = np.array([p[1] for p in best_final_cfg])
        angle_arr2 = np.array([p[2] for p in best_final_cfg])
        
        cur_score2 = best_final_side
        T2 = 0.05
        T2_end = 0.0005
        alpha2 = 0.998
        
        for step in range(3000):
            if time.time() - t0 > TIME_LIMIT - 8:
                break
            
            idx = random.randrange(N)
            old_x, old_y, old_a = cx_arr2[idx], cy_arr2[idx], angle_arr2[idx]
            
            cx_arr2[idx] += random.gauss(0, 0.03)
            cy_arr2[idx] += random.gauss(0, 0.03)
            angle_arr2[idx] += random.gauss(0, 0.5)
            
            valid = _is_valid_fast(cx_arr2, cy_arr2, angle_arr2)
            
            if valid:
                new_score = _quick_score(cx_arr2, cy_arr2, angle_arr2)
                if new_score < cur_score2 or (T2 > T2_end and random.random() < math.exp(-(new_score - cur_score2) / T2)):
                    cur_score2 = new_score
                    if new_score < best_final_side:
                        best_final_side = new_score
                        best_final_cfg = [[cx_arr2[i], cy_arr2[i], angle_arr2[i]] for i in range(N)]
                        best_final_center = (0.0, 0.0)
                        best_final_angle = 0.0
                else:
                    cx_arr2[idx], cy_arr2[idx], angle_arr2[idx] = old_x, old_y, old_a
            else:
                cx_arr2[idx], cy_arr2[idx], angle_arr2[idx] = old_x, old_y, old_a
            
            T2 = max(T2_end, T2 * alpha2)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        cfg = make_lattice_initial()
        scale = 1.3
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
