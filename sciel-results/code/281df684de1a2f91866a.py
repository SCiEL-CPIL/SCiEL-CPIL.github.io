import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_rad):
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hex(px, py, cx, cy, side, angle_rad):
    dx, dy = px - cx, py - cy
    ca, sa = np.cos(-angle_rad), np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    for k in range(6):
        a1 = k * np.pi / 3
        a2 = (k + 1) * np.pi / 3
        v1x = side * np.cos(a1)
        v1y = side * np.sin(a1)
        v2x = side * np.cos(a2)
        v2y = side * np.sin(a2)
        ex = v2x - v1x
        ey = v2y - v1y
        nx, ny = -ey, ex
        if nx * v1x + ny * v1y > 0:
            nx, ny = -nx, -ny
        if (lx - v1x) * nx + (ly - v1y) * ny < -1e-10:
            return False
    return True

def hex_inside_hex(verts_inner, cx, cy, side, angle_rad):
    for v in verts_inner:
        if not point_in_hex(v[0], v[1], cx, cy, side, angle_rad):
            return False
    return True

def point_in_poly(p, poly):
    sign = 0
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i+1)%n]
        cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        if cross > 1e-10:
            if sign == 0: sign = 1
            elif sign == -1: return False
        elif cross < -1e-10:
            if sign == 0: sign = -1
            elif sign == 1: return False
    return True

def seg_intersect(p1, p2, p3, p4):
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def hexes_overlap(v1, v2):
    for i in range(6):
        if point_in_poly(v1[i], v2) or point_in_poly(v2[i], v1):
            return True
    for i in range(6):
        for j in range(6):
            if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def objective(params):
    S = params[-1]
    if S <= 0:
        return 1e6
    inner_verts = []
    for i in range(N):
        ix = params[i*3]
        iy = params[i*3+1]
        ia = np.radians(params[i*3+2])
        verts = hex_vertices(ix, iy, 1.0, ia)
        inner_verts.append(verts)
    for i in range(N):
        if not hex_inside_hex(inner_verts[i], 0, 0, S, 0):
            return 1e6 + S
    for i in range(N):
        for j in range(i+1, N):
            if hexes_overlap(inner_verts[i], inner_verts[j]):
                return 1e6 + S
    return S

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Multiple initial configurations
    configs = []
    
    # Config 1: 1+6+5 hexagonal
    init1 = [0, 0, 0]
    for k in range(6):
        a = k * np.pi / 3
        init1.extend([np.sqrt(3)*np.cos(a), np.sqrt(3)*np.sin(a), 0])
    for k in [0, 1, 3, 4, 5]:
        a = k * np.pi / 3
        init1.extend([2*np.sqrt(3)*np.cos(a), 2*np.sqrt(3)*np.sin(a), 0])
    init1.append(10.0)
    configs.append(init1)
    
    # Config 2: Linear-ish arrangement
    init2 = []
    for k in range(12):
        init2.extend([k * np.sqrt(3), 0, 0])
    init2.append(20.0)
    configs.append(init2)
    
    best_params = None
    best_obj = float('inf')
    
    for init in configs:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        try:
            result = minimize(objective, init, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-3, 'fatol': 1e-3,
                                   'adaptive': True, 'maxfev': 5000})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: return initial seed-like result
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    S = best_params[-1]
    inner = []
    for i in range(N):
        inner.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
    
    return inner, (0.0, 0.0), S, 0.0
