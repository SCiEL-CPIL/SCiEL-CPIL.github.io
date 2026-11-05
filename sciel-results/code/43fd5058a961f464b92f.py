"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
TIME_LIMIT = 147.0  # safety margin below 150s

# Precompute unit hexagon vertices and normals for angle 0
_UNIT_VERTS = np.array([[math.cos(math.radians(60*k)), math.sin(math.radians(60*k))] for k in range(6)])
_UNIT_NORMALS = np.array([[math.cos(math.radians(30+60*k)), math.sin(math.radians(30+60*k))] for k in range(6)])

def _rotate_mat(angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa], [sa, ca]])

def _hex_vertices_arr(cx, cy, angle_deg):
    R = _rotate_mat(angle_deg)
    return cx + cy * np.array([1,1]) + np.dot(_UNIT_VERTS, R.T)

def _hex_normals_arr(angle_deg):
    R = _rotate_mat(angle_deg)
    return np.dot(_UNIT_NORMALS, R.T)

def _sat_overlap(vertsA, vertsB, normsA, normsB, tol=1e-9):
    """SAT: return True if overlapping."""
    axes = np.vstack([normsA, normsB])
    projA = vertsA @ axes.T
    projB = vertsB @ axes.T
    minA, maxA = projA.min(axis=1), projA.max(axis=1)
    minB, maxB = projB.min(axis=1), projB.max(axis=1)
    # Overlap on all axes means intersect
    if np.any(maxA <= minB + tol) or np.any(maxB <= minA + tol):
        return False
    return True

def _is_valid_fast(cfg):
    """Fast validity: all pairs disjoint via SAT."""
    m = len(cfg)
    verts = [np.array(_hex_vertices_arr(p[0], p[1], p[2])) for p in cfg]
    norms = [_hex_normals_arr(p[2]) for p in cfg]
    for i in range(m):
        for j in range(i+1, m):
            if _sat_overlap(verts[i], verts[j], norms[i], norms[j]):
                return False
    return True

def _outer_side(cfg, center, angle_deg):
    """Min side length of outer hexagon containing all inner hexagons."""
    apothem = math.sqrt(3) / 2.0
    R = _rotate_mat(angle_deg)
    # Outer hexagon edge normals
    outer_norms = np.dot(_UNIT_NORMALS, R.T)
    cx, cy = center
    max_need = 0.0
    for (px, py, pa) in cfg:
        v = np.array(_hex_vertices_arr(px, py, pa))
        rel = v - np.array([cx, cy])
        proj = rel @ outer_norms.T
        max_need = max(max_need, proj.max() / apothem)
    return max_need

def _smooth_outer_side(cfg, center, angle_deg, beta=20.0):
    """Smooth log-sum-exp of outer side."""
    apothem = math.sqrt(3) / 2.0
    R = _rotate_mat(angle_deg)
    outer_norms = np.dot(_UNIT_NORMALS, R.T)
    cx, cy = center
    vals = []
    for (px, py, pa) in cfg:
        v = np.array(_hex_vertices_arr(px, py, pa))
        rel = v - np.array([cx, cy])
        proj = rel @ outer_norms.T
        vals.extend(proj / apothem)
    vals = np.array(vals)
    mx = vals.max()
    return mx + (1.0/beta) * math.log(np.sum(np.exp(beta*(vals - mx))))

def _width_dist(p1, p2):
    """Width-based separation distance (negative if overlapping)."""
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    d = math.hypot(dx, dy)
    if d < 1e-10:
        return 0.0
    nx, ny = dx/d, dy/d
    # Support of hex1 toward hex2
    R1 = _rotate_mat(p1[2])
    s1 = (_UNIT_VERTS @ R1.T).dot([nx, ny]).max()
    R2 = _rotate_mat(p2[2])
    s2 = (_UNIT_VERTS @ R2.T).dot([-nx, -ny]).max()
    return d - s1 - s2

def _collision_penalty(cfg, beta=25.0):
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _width_dist(cfg[i], cfg[j])
            if d < 0:
                pen += d * d
    return pen

def _smooth_obj(v, beta=20.0):
    cfg = [[v[3*i], v[3*i+1], v[3*i+2]] for i in range(N)]
    center = (v[3*N], v[3*N+1])
    angle = v[3*N+2]
    return _smooth_outer_side(cfg, center, angle, beta=beta) + 50.0 * _collision_penalty(cfg)

def _flatten(cfg, center, angle):
    v = []
    for p in cfg:
        v.extend([p[0], p[1], p[2]])
    v.extend([center[0], center[1], angle])
    return np.array(v, dtype=float)

def _unflatten(v):
    cfg = [[v[3*i], v[3*i+1], v[3*i+2]] for i in range(N)]
    center = (v[3*N], v[3*N+1])
    angle = v[3*N+2]
    return cfg, center, angle

def make_honeycomb():
    """Honeycomb cluster: 1 center + 6 ring1 + 5 ring2."""
    pts = [[0.0, 0.0, 0.0]]
    R1 = math.sqrt(3)
    for i in range(6):
        th = 2*math.pi*i/6
        pts.append([R1*math.cos(th), R1*math.sin(th), 0.0])
    R2 = 2*math.sqrt(3)
    for i in range(5):
        th = 2*math.pi*(i+0.5)/5
        pts.append([R2*math.cos(th), R2*math.sin(th), 0.0])
    return pts

def make_hex_lattice():
    """3x4 hex lattice."""
    pts = []
    rh = math.sqrt(3)
    cw = 2.0
    for r in range(3):
        for c in range(4):
            x = c*cw + (cw/2 if r%2 else 0.0)
            y = r*rh
            pts.append([x - 3.0, y - rh, 0.0])
    return pts

def make_ring():
    pts = []
    R = 2.5
    for i in range(N):
        th = 2*math.pi*i/N
        pts.append([R*math.cos(th), R*math.sin(th), math.degrees(th)])
    return pts

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    def full_score(cfg, center, angle):
        if not _is_valid_fast(cfg):
            return None
        return _outer_side(cfg, center, angle)

    # Initial configs
    inits = [make_honeycomb(), make_hex_lattice(), make_ring()]
    best_score = float('inf')
    best_cfg = None
    best_center = [0.0, 0.0]
    best_angle = 0.0

    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0

    if best_cfg is None:
        best_cfg = make_honeycomb()
        best_score = 10.0

    # Simulated annealing
    T = 0.6
    T_end = 0.002
    alpha = 0.9998
    sa_limit = TIME_LIMIT - 35.0
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score

    while time.time() - t0 < sa_limit:
        r = random.random()
        if r < 0.45:
            idx = random.randrange(N)
            ox, oy = cfg[idx][0], cfg[idx][1]
            step = 0.15 * (1.0 + 0.5*T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.65:
            idx = random.randrange(N)
            oa = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 4.0 + 3.0*T)
        elif r < 0.75:
            ocx, ocy = center[0], center[1]
            center[0] += random.gauss(0, 0.08*(1.0+0.5*T))
            center[1] += random.gauss(0, 0.08*(1.0+0.5*T))
        elif r < 0.85:
            oa = angle
            angle += random.gauss(0, 2.0 + 2.0*T)
        else:
            i1, i2 = random.sample(range(N), 2)
            ox1, oy1 = cfg[i1][0], cfg[i1][1]
            ox2, oy2 = cfg[i2][0], cfg[i2][1]
            step = 0.12*(1.0+0.5*T)
            cfg[i1][0] += random.gauss(0, step)
            cfg[i1][1] += random.gauss(0, step)
            cfg[i2][0] += random.gauss(0, step)
            cfg[i2][1] += random.gauss(0, step)

        ns = full_score(cfg, tuple(center), angle)
        accept = False
        if ns is not None:
            if ns < cur_score:
                accept = True
                cur_score = ns
                if ns < best_score:
                    best_score = ns
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            elif T > T_end and random.random() < math.exp(-(ns - cur_score)/T):
                accept = True
                cur_score = ns

        if not accept:
            if r < 0.45:
                cfg[idx][0], cfg[idx][1] = ox, oy
            elif r < 0.65:
                cfg[idx][2] = oa
            elif r < 0.75:
                center[0], center[1] = ocx, ocy
            elif r < 0.85:
                angle = oa
            else:
                cfg[i1][0], cfg[i1][1] = ox1, oy1
                cfg[i2][0], cfg[i2][1] = ox2, oy2

        T = max(T_end, T * alpha)

    # Gradient polish stages
    for beta in [15.0, 30.0, 60.0]:
        remaining = TIME_LIMIT - (time.time() - t0)
        if remaining < 8.0:
            break
        try:
            v0 = _flatten(best_cfg, best_center, best_angle)
            def obj(v):
                if time.time() - t0 > TIME_LIMIT - 2.0:
                    return 1e6
                return _smooth_obj(v, beta=beta)
            res = minimize(obj, v0, method='L-BFGS-B',
                           options={'maxiter': 1500, 'maxfun': 8000, 'ftol': 1e-13, 'gtol': 1e-11})
            v_best = v0.copy()
            f_best = _smooth_obj(v0, beta=beta)
            if res.fun < f_best:
                v_best = res.x.copy()
                f_best = res.fun
            cfg_p, cen_p, ang_p = _unflatten(v_best)
            ps = full_score(cfg_p, cen_p, ang_p)
            if ps is not None and ps < best_score:
                best_score = ps
                best_cfg = cfg_p
                best_center = list(cen_p)
                best_angle = ang_p
        except Exception:
            pass

    # Final micro-refinement
    remaining = TIME_LIMIT - (time.time() - t0)
    if remaining > 3.0:
        for _ in range(800):
            if time.time() - t0 > TIME_LIMIT - 1.0:
                break
            cfg_try = [row[:] for row in best_cfg]
            idx = random.randrange(N)
            cfg_try[idx][0] += random.gauss(0, 0.005)
            cfg_try[idx][1] += random.gauss(0, 0.005)
            if random.random() < 0.3:
                cfg_try[idx][2] += random.gauss(0, 0.3)
            s = full_score(cfg_try, tuple(best_center), best_angle)
            if s is not None and s < best_score:
                best_score = s
                best_cfg = cfg_try

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
