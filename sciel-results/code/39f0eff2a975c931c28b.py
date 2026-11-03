import time
import numpy as np

N = 12
DEADLINE = 120.0
_t0 = time.time()

def _time_left():
    return DEADLINE - (time.time() - _t0)

def hex_vertices(cx, cy, rot_deg):
    a = np.deg2rad(rot_deg)
    ang = a + np.deg2rad(np.arange(6) * 60.0)
    return np.column_stack([cx + np.cos(ang), cy + np.sin(ang)])

def enclosing_hex_side(points, c, theta_deg):
    th = np.deg2rad(theta_deg)
    norms = th + np.deg2rad(np.arange(6) * 60.0)
    n = np.column_stack([np.cos(norms), np.sin(norms)])
    d = points - c
    proj = d @ n.T
    m = np.max(proj)
    return m / (np.sqrt(3.0) / 2.0)

def solve_lp(points, theta_deg):
    from scipy.optimize import linprog
    th = np.deg2rad(theta_deg)
    norms = th + np.deg2rad(np.arange(6) * 60.0)
    n = np.column_stack([np.cos(norms), np.sin(norms)])
    ap = np.sqrt(3.0) / 2.0
    A = []; b = []
    for p in points:
        for nk in n:
            A.append([-nk[0], -nk[1], -ap]); b.append(-float(p @ nk))
            A.append([ nk[0],  nk[1], -ap]); b.append(float(p @ nk))
    A = np.array(A); b = np.array(b)
    res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A, b_ub=b,
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    if res.success:
        return float(res.x[0]), float(res.x[1]), float(res.x[2])
    return None

def sat_overlap(p1, p2):
    for poly in (p1, p2):
        m = len(poly)
        for i in range(m):
            a = poly[i]; b = poly[(i + 1) % m]
            ex = b[0] - a[0]; ey = b[1] - a[1]
            nx = -ey; ny = ex
            pr1 = [v[0] * nx + v[1] * ny for v in p1]
            pr2 = [v[0] * nx + v[1] * ny for v in p2]
            if max(pr1) < min(pr2) or max(pr2) < min(pr1):
                return False
    return True

def config_valid(hexes):
    polys = [hex_vertices(h[0], h[1], h[2]) for h in hexes]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if sat_overlap(polys[i], polys[j]):
                return False
    return True

def build_lattice():
    axial = [(0, -2), (1, -2),
             (-1, -1), (0, -1), (1, -1),
             (-1, 0), (0, 0), (1, 0), (2, 0),
             (-1, 1), (0, 1), (1, 1)]
    hexes = []
    for q, r in axial:
        x = np.sqrt(3.0) * (q + r / 2.0)
        y = 1.5 * r
        hexes.append([float(x), float(y), 0.0])
    return hexes

def all_vertices(hexes):
    return np.vstack([hex_vertices(h[0], h[1], h[2]) for h in hexes])

def objective(x):
    n = N
    hexes = x[:3 * n].reshape(n, 3)
    cx, cy = x[3 * n], x[3 * n + 1]
    theta = x[3 * n + 2]
    pts = all_vertices(hexes)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(hexes[i, 0] - hexes[j, 0], hexes[i, 1] - hexes[j, 1])
            if d < np.sqrt(3.0):
                pen += (np.sqrt(3.0) - d) ** 2
    s = enclosing_hex_side(pts, np.array([cx, cy]), theta)
    return s + 1000.0 * pen

def _safe_result():
    hexes = build_lattice()
    pts = all_vertices(hexes)
    best = None
    for theta in np.linspace(0.0, 60.0, 181):
        r = solve_lp(pts, float(theta))
        if r is None:
            continue
        if best is None or r[2] < best[2]:
            best = (r[0], r[1], r[2], float(theta))
    if best is None:
        best = (0.0, 0.0, enclosing_hex_side(pts, np.array([0.0, 0.0]), 0.0), 0.0)
    cx, cy, s, theta = best
    inner = [[float(h[0]), float(h[1]), float(h[2])] for h in hexes]
    return inner, (float(cx), float(cy)), float(s), float(theta)

def construct():
    try:
        hexes = build_lattice()
        pts = all_vertices(hexes)
        best = None
        for theta in np.linspace(0.0, 60.0, 361):
            if _time_left() < 40:
                break
            r = solve_lp(pts, float(theta))
            if r is None:
                continue
            if best is None or r[2] < best[2]:
                best = (r[0], r[1], r[2], float(theta))
        if best is None:
            return _safe_result()
        cx, cy, s, theta = best
        if _time_left() > 50:
            x0 = np.concatenate([np.array([v for h in hexes for v in h]),
                                 [cx, cy, theta]])
            from scipy.optimize import minimize
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 250, 'xatol': 1e-4, 'fatol': 1e-5})
                if res is not None and res.fun < s:
                    x = res.x
                    new_hexes = [list(x[3 * i:3 * i + 3]) for i in range(N)]
                    ncx, ncy, ntheta = x[3 * N], x[3 * N + 1], x[3 * N + 2]
                    if config_valid(new_hexes):
                        ns = enclosing_hex_side(all_vertices(new_hexes),
                                                np.array([ncx, ncy]), ntheta)
                        if ns < s:
                            hexes = new_hexes
                            cx, cy, s, theta = ncx, ncy, ns, ntheta
            except Exception:
                pass
        if not config_valid(hexes):
            return _safe_result()
        inner = [[float(h[0]), float(h[1]), float(h[2])] for h in hexes]
        return inner, (float(cx), float(cy)), float(s), float(theta)
    except Exception:
        return _safe_result()
