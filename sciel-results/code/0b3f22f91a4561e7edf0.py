import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 146.0

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

def _sdf_hexagon_point(px, py, cx, cy, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx = px - cx
    dy = py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    apothem = math.sqrt(3) / 2.0
    
    min_dist = float('inf')
    for i in range(6):
        theta1 = math.radians(60 * i)
        theta2 = math.radians(60 * (i + 1))
        x1, y1 = math.cos(theta1), math.sin(theta1)
        x2, y2 = math.cos(theta2), math.sin(theta2)
        dx_e, dy_e = x2 - x1, y2 - y1
        len_sq = dx_e * dx_e + dy_e * dy_e
        if len_sq < 1e-12:
            d = math.hypot(lx - x1, ly - y1)
        else:
            t = ((lx - x1) * dx_e + (ly - y1) * dy_e) / len_sq
            t = max(0.0, min(1.0, t))
            px_e, py_e = x1 + t * dx_e, y1 + t * dy_e
            d = math.hypot(lx - px_e, ly - py_e)
        if d < min_dist:
            min_dist = d
    
    inside = True
    for k in range(6):
        theta = math.radians(60 * k + 90)
        nx, ny = math.cos(theta), math.sin(theta)
        if nx * lx + ny * ly > apothem + 1e-9:
            inside = False
            break
    
    return -min_dist if inside else min_dist

def _sdf_hexagon_hexagon(cx1, cy1, a1, cx2, cy2, a2):
    dist = math.hypot(cx1 - cx2, cy1 - cy2)
    if dist > 2.0 + 1e-9:
        return dist - 2.0
    
    d1 = _sdf_hexagon_point(cx2, cy2, cx1, cy1, a1)
    d2 = _sdf_hexagon_point(cx1, cy1, cx2, cy2, a2)
    
    if d1 < 0 and d2 < 0:
        return max(d1, d2)
    elif d1 < 0:
        return d1
    elif d2 < 0:
        return d2
    else:
        return max(d1, d2)

def _sdf_gradient_hexagon_hexagon(cx1, cy1, a1, cx2, cy2, a2, eps=1e-5):
    dx, dy, da1, da2 = 0.0, 0.0, 0.0, 0.0
    d = _sdf_hexagon_hexagon(cx1, cy1, a1, cx2, cy2, a2)
    
    if d > 1e-3:
        dist = math.hypot(cx1 - cx2, cy1 - cy2)
        if dist > 1e-6:
            dx = (cx1 - cx2) / dist
            dy = (cy1 - cy2) / dist
        return dx, dy, da1, da2, d
    
    d_xp = _sdf_hexagon_hexagon(cx1 + eps, cy1, a1, cx2, cy2, a2)
    d_xm = _sdf_hexagon_hexagon(cx1 - eps, cy1, a1, cx2, cy2, a2)
    d_yp = _sdf_hexagon_hexagon(cx1, cy1 + eps, a1, cx2, cy2, a2)
    d_ym = _sdf_hexagon_hexagon(cx1, cy1 - eps, a1, cx2, cy2, a2)
    d_a1p = _sdf_hexagon_hexagon(cx1, cy1, a1 + eps, cx2, cy2, a2)
    d_a1m = _sdf_hexagon_hexagon(cx1, cy1, a1 - eps, cx2, cy2, a2)
    d_a2p = _sdf_hexagon_hexagon(cx1, cy1, a1, cx2, cy2, a2 + eps)
    d_a2m = _sdf_hexagon_hexagon(cx1, cy1, a1, cx2, cy2, a2 - eps)
    
    dx = (d_xp - d_xm) / (2 * eps)
    dy = (d_yp - d_ym) / (2 * eps)
    da1 = (d_a1p - d_a1m) / (2 * eps)
    da2 = (d_a2p - d_a2m) / (2 * eps)
    
    return dx, dy, da1, da2, d

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

def _is_valid(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist = math.hypot(dx, dy)
            if dist < 2.0 - 1e-7:
                return False
            sdf = _sdf_hexagon_hexagon(cx_arr[i], cy_arr[i], angle_arr[i], cx_arr[j], cy_arr[j], angle_arr[j])
            if sdf < -1e-7:
                return False
    return True

def _objective_and_grad(params):
    n_params = 3 * N + 2
    xs = params[0:N]
    ys = params[N:2*N]
    angles = params[2*N:3*N]
    cx_out = params[3*N]
    cy_out = params[3*N + 1]
    
    a_out = 0.0
    apothem_unit = math.sqrt(3) / 2.0
    normals_out = []
    for k in range(6):
        na = a_out + math.radians(90 + 60 * k)
        normals_out.append((math.cos(na), math.sin(na)))
    
    max_need = 0.0
    for i in range(N):
        a = math.radians(angles[i])
        ca, sa = math.cos(a), math.sin(a)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = xs[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = ys[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals_out:
                val = (nx * (vx - cx_out) + ny * (vy - cy_out)) / apothem_unit
                if val > max_need:
                    max_need = val
    
    penalty = 0.0
    grad = np.zeros(n_params)
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq > (2.5) ** 2:
                continue
            
            sdf = _sdf_hexagon_hexagon(xs[i], ys[i], angles[i], xs[j], ys[j], angles[j])
            if sdf < 0:
                penalty += sdf * sdf
                gx, gy, ga1, ga2, _ = _sdf_gradient_hexagon_hexagon(xs[i], ys[i], angles[i], xs[j], ys[j], angles[j])
                grad[i] += 2 * sdf * gx
                grad[N + i] += 2 * sdf * gy
                grad[2*N + i] += 2 * sdf * ga1
                grad[j] -= 2 * sdf * gx
                grad[N + j] -= 2 * sdf * gy
                grad[2*N + j] -= 2 * sdf * ga2
    
    penalty_weight = 50.0
    obj = max_need + penalty_weight * penalty
    
    grad[3*N] = 0.0
    grad[3*N+1] = 0.0
    
    return obj, grad

def _quick_outer_side(xs, ys, angles, cx_out, cy_out):
    a_out = 0.0
    apothem_unit = math.sqrt(3) / 2.0
    normals_out = []
    for k in range(6):
        na = a_out + math.radians(90 + 60 * k)
        normals_out.append((math.cos(na), math.sin(na)))
    
    max_need = 0.0
    for i in range(N):
        a = math.radians(angles[i])
        ca, sa = math.cos(a), math.sin(a)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = xs[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = ys[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals_out:
                val = (nx * (vx - cx_out) + ny * (vy - cy_out)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

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
        if time.time() - t0 > TIME_LIMIT - 80:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid(cx_arr, cy_arr, angle_arr):
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

    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle

    # Multiple restarts with L-BFGS-B using SDF-based collision penalty
    for restart in range(4):
        if time.time() - t0 > TIME_LIMIT - 15:
            break
        
        cx_init = np.array([p[0] for p in best_final_cfg])
        cy_init = np.array([p[1] for p in best_final_cfg])
        angle_init = np.array([p[2] for p in best_final_cfg])
        
        if restart > 0:
            for i in range(N):
                cx_init[i] += random.gauss(0, 0.1)
                cy_init[i] += random.gauss(0, 0.1)
                angle_init[i] += random.gauss(0, 3.0)
        
        init_params = np.zeros(3 * N + 2)
        for i in range(N):
            init_params[i] = cx_init[i]
            init_params[N + i] = cy_init[i]
            init_params[2 * N + i] = angle_init[i]
        init_params[3 * N] = 0.0
        init_params[3 * N + 1] = 0.0
        
        try:
            result = minimize(
                _objective_and_grad,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 5000}
            )
            
            if result.fun < best_final_side or result.success:
                opt_params = result.x
                xs_opt = opt_params[0:N]
                ys_opt = opt_params[N:2*N]
                angles_opt = opt_params[2*N:3*N]
                cx_out_opt = opt_params[3*N]
                cy_out_opt = opt_params[3*N + 1]
                
                if _is_valid(xs_opt, ys_opt, angles_opt):
                    side_opt = _outer_hex_side(xs_opt, ys_opt, angles_opt, (cx_out_opt, cy_out_opt), 0.0)
                    if side_opt < best_final_side:
                        best_final_side = side_opt
                        best_final_cfg = [[xs_opt[i], ys_opt[i], angles_opt[i]] for i in range(N)]
                        best_final_center = (cx_out_opt, cy_out_opt)
                        best_final_angle = 0.0
        except Exception:
            pass
    
    # Refine outer hexagon center
    cx_best2 = np.array([p[0] for p in best_final_cfg])
    cy_best2 = np.array([p[1] for p in best_final_cfg])
    angle_best2 = np.array([p[2] for p in best_final_cfg])
    
    verts_best = [_hex_vertices(cx_best2[i], cy_best2[i], angle_best2[i]) for i in range(N)]
    
    for outer_angle in range(0, 30, 2):
        if time.time() - t0 > TIME_LIMIT - 3:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        cx_c, cy_c = best_final_center[0], best_final_center[1]
        for _iter in range(30):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
                    for (nx, ny) in normals:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            for dx in [-0.3, -0.15, -0.05, 0, 0.05, 0.15, 0.3]:
                for dy in [-0.3, -0.15, -0.05, 0, 0.05, 0.15, 0.3]:
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_best[i]:
                            for (nx, ny) in normals:
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
    
    if not _is_valid(cx_final, cy_final, angle_final):
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
