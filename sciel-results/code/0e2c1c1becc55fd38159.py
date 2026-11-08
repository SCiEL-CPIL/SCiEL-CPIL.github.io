import numpy as np
import time
from scipy.spatial import ConvexHull

N = 11

def hex_vertices(cx, cy, side, angle_deg=0.0):
    angle = np.radians(angle_deg)
    angles = angle + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def point_in_polygon(px, py, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygons_disjoint(poly1, poly2):
    for v in poly1:
        if point_in_polygon(v[0], v[1], poly2):
            return False
    for v in poly2:
        if point_in_polygon(v[0], v[1], poly1):
            return False
    def edges_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        p1, p2 = poly1[i], poly1[(i+1) % n1]
        for j in range(n2):
            p3, p4 = poly2[j], poly2[(j+1) % n2]
            if edges_intersect(p1, p2, p3, p4):
                return False
    return True

def hexagons_disjoint_check(inner_hexes):
    n = len(inner_hexes)
    polys = [hex_vertices(hx, hy, 1.0, ang) for hx, hy, ang in inner_hexes]
    for i in range(n):
        for j in range(i+1, n):
            if not polygons_disjoint(polys[i], polys[j]):
                return False
    return True

def min_outer_hexagon_fast(inner_hexes):
    all_points = np.vstack([hex_vertices(hx, hy, 1.0, ang) for hx, hy, ang in inner_hexes])
    cx0, cy0 = np.mean(all_points[:, 0]), np.mean(all_points[:, 1])
    best_side = float('inf')
    best_cx, best_cy, best_theta = cx0, cy0, 0.0
    for theta_deg in np.linspace(0, 180, 36, endpoint=False):
        theta = np.radians(theta_deg)
        thetas = theta + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
        cos_t, sin_t = np.cos(thetas), np.sin(thetas)
        def obj(params):
            cx, cy = params
            max_e = 0.0
            for k in range(6):
                projs = all_points[:, 0] * cos_t[k] + all_points[:, 1] * sin_t[k] - cx * cos_t[k] - cy * sin_t[k]
                max_e = max(max_e, np.max(projs), -np.min(projs))
            return max_e
        from scipy.optimize import minimize
        res = minimize(obj, [cx0, cy0], method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
        cx_o, cy_o = res.x
        side = 2.0 * res.fun / np.sqrt(3.0)
        if side < best_side:
            best_side, best_cx, best_cy, best_theta = side, cx_o, cy_o, theta_deg
    return best_cx, best_cy, best_side, best_theta

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    dx, dy = np.sqrt(3.0), 1.5
    positions = []
    for i in range(2):
        positions.append([(i - 0.5) * dx, 0.0, 0.0])
    for row in range(1, 4):
        for i in range(3):
            positions.append([(i - 1.0) * dx, row * dy, 0.0])
    inner = positions
    if not hexagons_disjoint_check(inner):
        inner = [[k * 2.0, 0.0, 0.0] for k in range(N)]
    cx, cy, side, theta = min_outer_hexagon_fast(inner)
    best_inner, best_side = inner[:], side
    np.random.seed(42)
    for _ in range(80):
        if time.time() > deadline - 15.0:
            break
        trial = [list(p) for p in best_inner]
        for i in range(len(trial)):
            trial[i][0] += np.random.normal(0, 0.15)
            trial[i][1] += np.random.normal(0, 0.15)
        if not hexagons_disjoint_check(trial):
            continue
        cx_t, cy_t, side_t, theta_t = min_outer_hexagon_fast(trial)
        if side_t < best_side:
            best_side, best_inner = side_t, trial
            cx, cy, theta = cx_t, cy_t, theta_t
    for _ in range(40):
        if time.time() > deadline - 5.0:
            break
        trial = [list(p) for p in best_inner]
        idx = np.random.randint(0, len(trial))
        trial[idx][2] += np.random.choice([-15, -10, -5, 5, 10, 15])
        if not hexagons_disjoint_check(trial):
            continue
        cx_t, cy_t, side_t, theta_t = min_outer_hexagon_fast(trial)
        if side_t < best_side:
            best_side, best_inner = side_t, trial
            cx, cy, theta = cx_t, cy_t, theta_t
    return best_inner, (cx, cy), best_side, theta

# EVOLVE-BLOCK-START
# construct is defined above
# EVOLVE-BLOCK-END
