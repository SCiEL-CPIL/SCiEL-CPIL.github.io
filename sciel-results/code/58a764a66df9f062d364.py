import numpy as np
import time

N = 12
DEADLINE = 145.0

def hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    R = side
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + R * np.cos(angles), cy + R * np.sin(angles)])

def point_in_hex(px, py, cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    r = side * np.sqrt(3) / 2
    for k in range(6):
        theta = a + k * np.pi / 3
        nx, ny = np.cos(theta), np.sin(theta)
        if (px - cx) * nx + (py - cy) * ny > r + 1e-9:
            return False
    return True

def min_enclosing_hex_side(points, angle_deg):
    a = np.radians(angle_deg)
    normals = np.array([[np.cos(a + k * np.pi / 3), np.sin(a + k * np.pi / 3)] for k in range(6)])
    dots = points @ normals.T
    f = dots.max(axis=0)
    g = dots.min(axis=0)
    
    from scipy.optimize import linprog
    A_ub = np.zeros((12, 3))
    b_ub = np.zeros(12)
    for k in range(6):
        A_ub[2*k] = [-normals[k, 0], -normals[k, 1], -1.0]
        b_ub[2*k] = -f[k]
        A_ub[2*k+1] = [normals[k, 0], normals[k, 1], -1.0]
        b_ub[2*k+1] = g[k]
    c_obj = np.array([0.0, 0.0, 1.0])
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')
    if res.success:
        cx, cy, r = res.x
        side = r * 2 / np.sqrt(3)
        return side, cx, cy
    else:
        cx, cy = points.mean(axis=0)
        max_r = 0
        for k in range(6):
            vals = (points - np.array([cx, cy])) @ normals[k]
            max_r = max(max_r, abs(vals).max())
        side = max_r * 2 / np.sqrt(3)
        return side, cx, cy

def get_all_vertices(inner):
    all_pts = []
    for x, y, ang in inner:
        all_pts.append(hex_vertices(x, y, 1.0, ang))
    return np.vstack(all_pts)

def check_no_overlap(inner):
    n = len(inner)
    verts_list = [hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for p in verts_list[i]:
                if point_in_hex(p[0], p[1], inner[j][0], inner[j][1], 1.0, inner[j][2]):
                    return False
            for p in verts_list[j]:
                if point_in_hex(p[0], p[1], inner[i][0], inner[i][1], 1.0, inner[i][2]):
                    return False
    return True

def make_config_1():
    d = 2.0
    inner = []
    for i in range(3):
        for j in range(4):
            x = (j - 1.5) * d
            y = (i - 1) * d * 0.866
            inner.append([x, y, 0.0])
    return inner

def make_config_2():
    inner = [[0.0, 0.0, 0.0]]
    d = 2.0
    for k in range(6):
        a = k * np.pi / 3
        inner.append([d * np.cos(a), d * np.sin(a), 0.0])
    d2 = 4.0
    for k in range(5):
        a = k * 2 * np.pi / 5 + np.pi/5
        inner.append([d2 * np.cos(a), d2 * np.sin(a), 0.0])
    return inner

def make_config_3():
    d = 2.0
    inner = []
    for i in range(4):
        for j in range(3):
            x = (j - 1) * d
            y = (i - 1.5) * d * 0.866
            if i % 2 == 1:
                x += d/2
            inner.append([x, y, 0.0])
    return inner

def local_refine(inner, outer_angle):
    from scipy.optimize import minimize
    
    def objective(params):
        inner_opt = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
        if not check_no_overlap(inner_opt):
            return 1e6
        all_pts = get_all_vertices(inner_opt)
        side, _, _ = min_enclosing_hex_side(all_pts, outer_angle)
        return side
    
    x0 = np.array([v for row in inner for v in row])
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 80, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
        if res.fun < 1e6:
            inner_opt = [[res.x[3*i], res.x[3*i+1], res.x[3*i+2]] for i in range(N)]
            return inner_opt, res.fun
    except Exception:
        pass
    return inner, None

def construct():
    t0 = time.time()
    configs = [make_config_1(), make_config_2(), make_config_3()]
    outer_angles = [0, 30, 15, 45, 20, 25, 10, 5, 35, 40, 50, 55, 60, 22, 28, 32]
    
    best_side = float('inf')
    best_result = None
    
    for inner in configs:
        if time.time() - t0 > DEADLINE:
            break
        for oa in outer_angles:
            if time.time() - t0 > DEADLINE:
                break
            all_pts = get_all_vertices(inner)
            side, center, angle = min_enclosing_hex_side(all_pts, oa)
            if not check_no_overlap(inner):
                continue
            if side < best_side:
                best_side = side
                best_result = (inner, center, side, angle)
        
        if time.time() - t0 > DEADLINE - 3:
            break
        best_oa = 0
        best_oa_side = float('inf')
        for oa in outer_angles:
            all_pts = get_all_vertices(inner)
            s, _, _ = min_enclosing_hex_side(all_pts, oa)
            if s < best_oa_side:
                best_oa_side = s
                best_oa = oa
        if best_oa_side < best_side:
            try:
                inner_opt, opt_side = local_refine(inner, best_oa)
                if opt_side is not None and opt_side < best_side:
                    all_pts = get_all_vertices(inner_opt)
                    side, center, angle = min_enclosing_hex_side(all_pts, best_oa)
                    if check_no_overlap(inner_opt):
                        best_side = side
                        best_result = (inner_opt, center, side, angle)
            except Exception:
                pass
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    return best_result
