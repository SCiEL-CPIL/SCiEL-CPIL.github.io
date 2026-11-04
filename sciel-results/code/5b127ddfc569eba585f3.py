import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
GR = (math.sqrt(5.0) - 1.0) / 2.0

# Vertex directions of a unit flat-top hexagon (vertex-angle 0)
_VERT0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

# Topological template: 1 center + 6 inner ring + 4 outer ring
_BASE = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [-0.5, SQRT3 / 2],
    [-0.5, -SQRT3 / 2],
    [1.0, 0.0],
    [-0.5, SQRT3 / 2],
    [-0.5, -SQRT3 / 2],
    [0.5, SQRT3 / 2],
    [0.5, -SQRT3 / 2],
    [-0.5, SQRT3 / 2],
    [-0.5, -SQRT3 / 2],
])
_RING = np.zeros(N)
_RING[1:7] = 1.0
_RING[7:11] = 2.0


def _rot(v, c, s):
    return np.column_stack([c * v[:, 0] - s * v[:, 1],
                            s * v[:, 0] + c * v[:, 1]])


def _normals(g):
    """Face normals of outer hexagon with vertex-angle g: at g+30+60k."""
    out = np.empty((6, 2))
    for k in range(6):
        a = math.radians(g + 30 + k * 60)
        out[k] = [math.cos(a), math.sin(a)]
    return out


def _verts_from_params(params):
    r1, r2, g = params[0], params[1], params[2]
    cg, sg = math.cos(g), math.sin(g)
    scale = np.where(_RING == 1.0, r1, np.where(_RING == 2.0, r2, 1.0))
    centers = _BASE * scale
    centers = _rot(centers, cg, sg)
    micro = params[3:] if len(params) == N else np.zeros(N)
    allv = np.empty((N * 6, 2))
    for i in range(N):
        a = g + micro[i]
        allv[i * 6:(i + 1) * 6] = _rot(_VERT0, math.cos(a), math.sin(a)) + centers[i]
    return allv


def _fast_side(params):
    allv = _verts_from_params(params)
    nm = _normals(params[2])
    dots = allv @ nm.T
    return 2.0 * float(np.max(dots)) / SQRT3


def _sep_poly(v1, v2):
    for v in v1:
        sgn = 0
        for k in range(6):
            a, b = v1[k], v1[(k + 1) % 6]
            cr = (b[0] - a[0]) * (v[1] - a[1]) - (b[1] - a[1]) * (v[0] - a[0])
            if cr > 1e-12:
                if sgn < 0:
                    return 0.0
                sgn = 1
            elif cr < -1e-12:
                if sgn > 0:
                    return 0.0
                sgn = -1
    for v in v2:
        sgn = 0
        for k in range(6):
            a, b = v2[k], v2[(k + 1) % 6]
            cr = (b[0] - a[0]) * (v[1] - a[1]) - (b[1] - a[1]) * (v[0] - a[0])
            if cr > 1e-12:
                if sgn < 0:
                    return 0.0
                sgn = 1
            elif cr < -1e-12:
                if sgn > 0:
                    return 0.0
                sgn = -1
    md = float('inf')
    for i in range(6):
        a1, a2 = v1[i], v1[(i + 1) % 6]
        for j in range(6):
            b1, b2 = v2[j], v2[(j + 1) % 6]
            d1 = a2 - a1
            d2 = b2 - b1
            r = a1 - b1
            aa, ee = float(np.dot(d1, d1)), float(np.dot(d2, d2))
            ff, cc, bb = float(np.dot(d2, r)), float(np.dot(d1, r)), float(np.dot(d1, d2))
            den = aa * ee - bb * bb
            if den > 1e-12:
                s = (bb * ff - cc * ee) / den
                s = max(0.0, min(1.0, s))
                t = (bb * s + ff) / ee
                if t < 0.0:
                    t = 0.0
                    s = max(0.0, min(1.0, cc / aa)) if aa > 1e-12 else 0.0
                elif t > 1.0:
                    t = 1.0
                    s = max(0.0, min(1.0, (bb + ff) / aa)) if aa > 1e-12 else 0.0
                d = (a1 + s * d1) - (b1 + t * d2)
                md = min(md, float(np.dot(d, d)))
            elif ee > 1e-12:
                t = max(0.0, min(1.0, ff / ee))
                d = a1 - (b1 + t * d2)
                md = min(md, float(np.dot(d, d)))
    return math.sqrt(md)


def _overlap_penalty(params, w=500.0, thresh=0.02):
    allv = _verts_from_params(params)
    pen = 0.0
    for i in range(N):
        v1 = allv[i * 6:(i + 1) * 6]
        for j in range(i + 1, N):
            v2 = allv[j * 6:(j + 1) * 6]
            sep = _sep_poly(v1, v2)
            if sep < thresh:
                pen += w * (thresh - sep) ** 2
    return pen


def _objective(params):
    return _fast_side(params) + _overlap_penalty(params)


def _golden1d(f, lo, hi, iters=40):
    a, b = lo, hi
    c = b - GR * (b - a)
    d = a + GR * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - GR * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + GR * (b - a)
            fd = f(d)
    x = 0.5 * (a + b)
    return x, f(x)


def _is_valid(params, side):
    if side < _fast_side(params) - 1e-9:
        return False
    allv = _verts_from_params(params)
    for i in range(N):
        v1 = allv[i * 6:(i + 1) * 6]
        for j in range(i + 1, N):
            v2 = allv[j * 6:(j + 1) * 6]
            if _sep_poly(v1, v2) < 1e-9:
                return False
    return True


def _seed_rows():
    rows = [[0.0, 0.0, 0.0]]
    for k in range(6):
        a = math.radians(k * 60)
        rows.append([SQRT3 * math.cos(a), SQRT3 * math.sin(a), 0.0])
    for k in range(4):
        a = math.radians(30 + k * 60)
        rows.append([3.0 * math.cos(a), 3.0 * math.sin(a), 0.0])
    return rows


def construct():
    start_time = time.time()
    DEADLINE = 145.0
    MARGIN = 5.0

    r1, r2, g0 = SQRT3, 3.0, 0.0
    seed_allv = _verts_from_params(np.array([r1, r2, g0]))
    seed_side = _fast_side(np.array([r1, r2, g0]))
    best_params = np.array([r1, r2, g0])
    best_side = seed_side

    def _maybe(params, side):
        nonlocal best_params, best_side
        if side >= best_side - 1e-15:
            return
        if not _is_valid(params, side):
            return
        best_params = params.copy()
        best_side = side

    # Phase 1: reduced 3D search (r1, r2, g)
    def obj3(p):
        return _fast_side(np.array([p[0], p[1], p[2]]))

    starts = [np.array([r1, r2, g0])]
    for k in range(1, 6):
        starts.append(np.array([r1 + 0.05 * k * (1 if k % 2 else -1),
                                r2 + 0.08 * k * (1 if k % 2 else -1),
                                (g0 + k * 0.15) % (2 * math.pi)]))
    for s in starts:
        if time.time() > DEADLINE - MARGIN - 25:
            break
        try:
            res = minimize(obj3, s, method='Powell',
                           options={'maxiter': 2000, 'ftol': 1e-14, 'xtol': 1e-14, 'disp': False})
            cand = np.array([res.x[0], res.x[1], res.x[2]])
            _maybe(cand, _fast_side(cand))
        except Exception:
            pass

    # Phase 2: add micro-rotations
    if time.time() < DEADLINE - MARGIN - 15:
        full = np.concatenate([best_params, np.zeros(N)])
        try:
            res = minimize(_objective, full, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            _maybe(res.x, _fast_side(res.x))
        except Exception:
            pass

    # Phase 3: golden-section coordinate descent polish
    if time.time() < DEADLINE - MARGIN - 3:
        params = best_params.copy()
        cur = _fast_side(params)
        for _ in range(60):
            if time.time() > DEADLINE - MARGIN - 1:
                break
            improved = False
            for i in range(len(params)):
                p = params[i]
                half = 0.02
                if i == 1:
                    half = 0.03
                if i == 2:
                    def f1(t, _p0=params[0], _p1=params[1]):
                        return _fast_side(np.array([_p0, _p1, t]))
                else:
                    def f1(t, _i=i):
                        q = params.copy()
                        q[_i] = t
                        return _fast_side(q)
                x, xv = _golden1d(f1, p - half, p + half, iters=40)
                if xv < cur - 1e-15 and abs(x - p) > 1e-14:
                    params[i] = x
                    cur = xv
                    improved = True
            if not improved:
                break
        _maybe(params, _fast_side(params))

    final_side = _fast_side(best_params)
    if not _is_valid(best_params, final_side):
        best_params = np.array([r1, r2, g0])
        final_side = seed_side

    r1f, r2f, gf = best_params[0], best_params[1], best_params[2]
    scale = np.where(_RING == 1.0, r1f, np.where(_RING == 2.0, r2f, 1.0))
    centers = _rot(_BASE * scale, math.cos(gf), math.sin(gf))
    micro = best_params[3:] if len(best_params) == N else np.zeros(N)

    inner_rows = []
    for i in range(N):
        inner_rows.append([float(centers[i, 0]), float(centers[i, 1]),
                           float(math.degrees(gf + micro[i]))])

    outer_angle_deg = float(math.degrees(gf))
    side = float(final_side)

    if side > seed_side:
        side = float(seed_side)
        inner_rows = _seed_rows()
        outer_angle_deg = 0.0

    return inner_rows, (0.0, 0.0), side, outer_angle_deg


if __name__ == "__main__":
    _ = construct()
