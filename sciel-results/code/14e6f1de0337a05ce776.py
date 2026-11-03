import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 148.0


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation (degrees)."""
    ar = math.radians(angle_deg)
    return [(cx + math.cos(ar + math.pi * k / 3.0),
             cy + math.sin(ar + math.pi * k / 3.0)) for k in range(6)]


def _pt_in_convex(pt, poly, eps=1e-9):
    px, py = pt
    for i in range(6):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % 6]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross > eps:
            return False
    return True


def _seg_intersect(p1, p2, p3, p4, eps=1e-9):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    if abs(d1) <= eps and _on_seg(p3, p1, p4, eps): return True
    if abs(d2) <= eps and _on_seg(p3, p2, p4, eps): return True
    if abs(d3) <= eps and _on_seg(p1, p3, p2, eps): return True
    if abs(d4) <= eps and _on_seg(p1, p4, p2, eps): return True
    return False


def _on_seg(p, q, r, eps=1e-9):
    if (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and
        min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps and
        abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])) <= eps):
        return True
    return False


def _pt_seg_dist2(pt, a, b):
    ax, ay = a; bx, by = b; px, py = pt
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return (px - ax)**2 + (py - ay)**2
    t = ((px - ax)*dx + (py - ay)*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return (px - cx)**2 + (py - cy)**2


def _min_dist_poly_fast(poly1, poly2):
    """Minimum distance between two convex hexagons (0 if overlapping)."""
    minx1 = min(p[0] for p in poly1); maxx1 = max(p[0] for p in poly1)
    miny1 = min(p[1] for p in poly1); maxy1 = max(p[1] for p in poly1)
    minx2 = min(p[0] for p in poly2); maxx2 = max(p[0] for p in poly2)
    miny2 = min(p[1] for p in poly2); maxy2 = max(p[1] for p in poly2)

    if maxx1 < minx2 - 1e-9 or maxx2 < minx1 - 1e-9 or maxy1 < miny2 - 1e-9 or maxy2 < miny1 - 1e-9:
        best = float('inf')
        for a in poly1:
            for b in poly2:
                d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                if d < best:
                    best = d
        return math.sqrt(best)

    for v in poly1:
        if _pt_in_convex(v, poly2):
            return 0.0
    for v in poly2:
        if _pt_in_convex(v, poly1):
            return 0.0

    for i in range(6):
        for j in range(6):
            if _seg_intersect(poly1[i], poly1[(i+1) % 6], poly2[j], poly2[(j+1) % 6]):
                return 0.0

    best = float('inf')
    for a in poly1:
        for b in poly2:
            d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d < best:
                best = d
        for i in range(6):
            d = _pt_seg_dist2(a, poly2[i], poly2[(i+1) % 6])
            if d < best:
                best = d
    for b in poly2:
        for i in range(6):
            d = _pt_seg_dist2(b, poly1[i], poly1[(i+1) % 6])
            if d < best:
                best = d
    return math.sqrt(best)


def _check_disjoint(inner_hexes, tol=1e-6):
    """Check all pairs are disjoint (non-overlapping)."""
    n = len(inner_hexes)
    polys = [_hex_vertices(h[0], h[1], h[2]) for h in inner_hexes]
    for i in range(n):
        for j in range(i + 1, n):
            d = _min_dist_poly_fast(polys[i], polys[j])
            if d < tol:
                return False
    return True


def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """LP: minimum side length of outer hexagon with given angle containing all inner hex vertices."""
    all_verts = []
    for ix, iy, ia in inner_hexes:
        all_verts.extend(_hex_vertices(ix, iy, ia))
    
    if not all_verts:
        return 0.0, (0.0, 0.0), 0.0

    all_verts = np.array(all_verts)
    
    c = [0.0, 0.0, 1.0]
    A_ub = []
    b_ub = []

    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx, ny = math.cos(a), math.sin(a)
        for vx, vy in all_verts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
        for vx, vy in all_verts:
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                         bounds=[(None, None), (None, None), (0, None)],
                         method='highs')
        if result.success:
            cx, cy, t = result.x
            return cx, cy, t / (math.sqrt(3) / 2)
    except:
        pass
        
    cx = sum(v[0] for v in all_verts) / len(all_verts)
    cy = sum(v[1] for v in all_verts) / len(all_verts)
    req = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx, ny = math.cos(a), math.sin(a)
        for vx, vy in all_verts:
            d = (vx - cx) * nx + (vy - cy) * ny
            req = max(req, abs(d))
    return cx, cy, req / (math.sqrt(3) / 2)


def _build_lattice_positions(scale=1.001):
    """Build hexagonal lattice positions for N hexagons."""
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = (i * sqrt3 + j * sqrt3 / 2) * scale
            y = j * 1.5 * scale
            d2 = x * x + y * y
            lattice_points.append((d2, x, y))
    lattice_points.sort(key=lambda t: t[0])

    seen = set()
    positions = []
    for d2, x, y in lattice_points:
        key = (round(x, 8), round(y, 8))
        if key in seen:
            continue
        seen.add(key)
        positions.append((x, y))
        if len(positions) == N:
            break

    if len(positions) < N:
        positions = [(k * sqrt3 * scale, 0.0) for k in range(N)]
    return positions


def _objective(params, positions_ref, angle_fixed=None):
    """Objective: outer hex side length + smooth penalty for overlap."""
    if time.time() > _DEADLINE - 5.0:
        return 9999.0

    n = N
    inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(n)]

    polys = [_hex_vertices(h[0], h[1], h[2]) for h in inner]
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = _min_dist_poly_fast(polys[i], polys[j])
            if d < 0.01:
                penalty += (0.01 - d) ** 2

    if penalty > 1e-8:
        cx = sum(h[0] for h in inner) / n
        cy = sum(h[1] for h in inner) / n
        maxr = max(math.sqrt((h[0]-cx)**2 + (h[1]-cy)**2) + 1.0 for h in inner)
        outer = maxr / (math.sqrt(3)/2)
        return outer + 500.0 * penalty

    best_s = float('inf')
    angles_to_try = [0.0, 15.0, 30.0, 45.0] if angle_fixed is None else [angle_fixed]
    for ang in angles_to_try:
        _, _, s = _min_enclosing_hex_side_lp(inner, ang)
        if s < best_s:
            best_s = s

    return best_s


def _optimize_from_start(x0, max_iter=200, method='L-BFGS-B'):
    """Run optimization from a starting point."""
    bounds = []
    for i in range(N):
        bounds.append((None, None))
        bounds.append((None, None))
        bounds.append((0.0, 360.0))

    try:
        res = minimize(_objective, x0, method=method, bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False})
        return res
    except:
        return None


def construct():
    global _DEADLINE

    if time.time() > _DEADLINE - 3.0:
        inner = [[0.0, 0.0, 30.0]] * N
        return inner, (0.0, 0.0), 50.0, 0.0

    sqrt3 = math.sqrt(3)
    scale = 1.001

    # Build initial lattice configuration
    positions = _build_lattice_positions(scale)
    initial_inner = [[px, py, 30.0] for px, py in positions]
    
    x0 = []
    for h in initial_inner:
        x0.extend(h)

    best_inner = initial_inner
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Initial evaluation
    for ang in [0.0, 15.0, 30.0, 45.0]:
        cx, cy, s = _min_enclosing_hex_side_lp(initial_inner, ang)
        if s < best_side:
            best_side = s
            best_center = (cx, cy)
            best_angle = ang

    # Multi-start optimization with different angle initializations
    if time.time() < _DEADLINE - 30.0:
        for start_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
            if time.time() > _DEADLINE - 25.0:
                break

            # Try different lattice scales and offsets
            for scale_try in [1.001, 1.002, 1.003]:
                if time.time() > _DEADLINE - 20.0:
                    break

                positions_try = _build_lattice_positions(scale_try)
                inner_try = [[px, py, start_angle] for px, py in positions_try]
                x0_try = []
                for h in inner_try:
                    x0_try.extend(h)

                res = _optimize_from_start(x0_try, max_iter=150)
                if res is not None and (res.success or res.fun < best_side):
                    x_opt = res.x
                    new_inner = [[x_opt[3*i], x_opt[3*i+1], x_opt[3*i+2]] for i in range(N)]
                    
                    if _check_disjoint(new_inner, tol=1e-5):
                        for ang in [0.0, 15.0, 30.0, 45.0]:
                            cx, cy, s = _min_enclosing_hex_side_lp(new_inner, ang)
                            if s < best_side:
                                best_side = s
                                best_center = (cx, cy)
                                best_angle = ang
                                best_inner = new_inner

    # Nelder-Mead polish on best found
    if time.time() < _DEADLINE - 15.0:
        x0_polish = []
        for h in best_inner:
            x0_polish.extend(h)
        
        try:
            res = minimize(_objective, x0_polish, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < best_side:
                x_opt = res.x
                new_inner = [[x_opt[3*i], x_opt[3*i+1], x_opt[3*i+2]] for i in range(N)]
                if _check_disjoint(new_inner, tol=1e-5):
                    for ang in [0.0, 15.0, 30.0, 45.0]:
                        cx, cy, s = _min_enclosing_hex_side_lp(new_inner, ang)
                        if s < best_side:
                            best_side = s
                            best_center = (cx, cy)
                            best_angle = ang
                            best_inner = new_inner
        except:
            pass

    # Fine angle sweep
    if time.time() < _DEADLINE - 8.0:
        for k in range(80):
            ang = (best_angle + 30.0 * (k / 79.0 - 0.5)) % 60
            cx, cy, s = _min_enclosing_hex_side_lp(best_inner, ang)
            if s < best_side:
                best_side = s
                best_center = (cx, cy)
                best_angle = ang

    # Even finer sweep
    if time.time() < _DEADLINE - 5.0:
        for k in range(50):
            ang = (best_angle + 5.0 * (k / 49.0 - 0.5)) % 60
            cx, cy, s = _min_enclosing_hex_side_lp(best_inner, ang)
            if s < best_side:
                best_side = s
                best_center = (cx, cy)
                best_angle = ang

    # Final L-BFGS-B polish with fixed best angle
    if time.time() < _DEADLINE - 3.0:
        x0_final = []
        for h in best_inner:
            x0_final.extend(h)
        
        try:
            # Objective with fixed angle for tighter optimization
            def _obj_fixed_angle(params):
                if time.time() > _DEADLINE - 2.0:
                    return 9999.0
                n = N
                inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(n)]
                polys = [_hex_vertices(h[0], h[1], h[2]) for h in inner]
                penalty = 0.0
                for i in range(n):
                    for j in range(i + 1, n):
                        d = _min_dist_poly_fast(polys[i], polys[j])
                        if d < 0.01:
                            penalty += (0.01 - d) ** 2
                if penalty > 1e-8:
                    cx = sum(h[0] for h in inner) / n
                    cy = sum(h[1] for h in inner) / n
                    maxr = max(math.sqrt((h[0]-cx)**2 + (h[1]-cy)**2) + 1.0 for h in inner)
                    return maxr / (math.sqrt(3)/2) + 500.0 * penalty
                _, _, s = _min_enclosing_hex_side_lp(inner, best_angle)
                return s

            bounds = []
            for i in range(N):
                bounds.append((None, None))
                bounds.append((None, None))
                bounds.append((0.0, 360.0))

            res = minimize(_obj_fixed_angle, x0_final, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
            if res.success or res.fun < best_side:
                x_opt = res.x
                new_inner = [[x_opt[3*i], x_opt[3*i+1], x_opt[3*i+2]] for i in range(N)]
                if _check_disjoint(new_inner, tol=1e-5):
                    _, _, s = _min_enclosing_hex_side_lp(new_inner, best_angle)
                    if s < best_side:
                        best_side = s
                        best_center = (sum(h[0] for h in new_inner)/N, sum(h[1] for h in new_inner)/N)
                        best_inner = new_inner
        except:
            pass

    return best_inner, (best_center[0], best_center[1]), best_side, best_angle
