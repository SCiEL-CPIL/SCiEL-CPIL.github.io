import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 6.0
_TIME_DEADLINE = START_TIME + TIME_LIMIT - SAFETY_MARGIN

# ---------------------------------------------------------------------------
# Candidate topologies as (x, y, angle_deg) lists.  Each is a starting guess for
# the 11 inner hexagons; positions and rotations are then freely optimised.
# ---------------------------------------------------------------------------
def _star():
    pts = [(0.0, 0.0, 0.0)]
    for k in range(6):
        a = math.radians(60 * k)
        pts.append((math.sqrt(3) * math.cos(a), 1.5 * math.sin(a), 0.0))
    return pts

def _build():
    cands = []
    cands.append(_star())

    # 3x3 block (axial) + 2 extras
    def axial3x3():
        pts = []
        for q in range(3):
            for r in range(3):
                pts.append((math.sqrt(3) * (q + r / 2.0), 1.5 * r, 0.0))
        return pts
    b = axial3x3()
    b.append((3.0 * math.sqrt(3), 1.5 * math.sqrt(3) / 2.0, 0.0))
    b.append((-math.sqrt(3), 1.5 * math.sqrt(3) / 2.0, 0.0))
    cands.append(b)

    # 4x3 block (axial) - 1
    def axial4x3():
        pts = []
        for q in range(4):
            for r in range(3):
                if not (q == 0 and r == 0):
                    pts.append((math.sqrt(3) * (q + r / 2.0), 1.5 * r, 0.0))
        return pts
    cands.append(axial4x3())

    # T shape
    t = [(0.0, 0.0, 0.0)]
    for k in range(6):
        a = math.radians(60 * k)
        t.append((math.sqrt(3) * math.cos(a), 1.5 * math.sin(a), 0.0))
    for k in range(4):
        a = math.radians(60 * k)
        t.append((2 * math.sqrt(3) * math.cos(a), 3.0 * math.sin(a), 0.0))
    cands.append(t)

    # diamond-ish (axial)
    d = []
    for q, r in [(0, 0), (1, 0), (-1, 0), (0, 1), (1, 1), (-1, 1),
                 (0, 2), (1, 2), (-1, -1), (0, -1), (1, -1)]:
        d.append((math.sqrt(3) * (q + r / 2.0), 1.5 * r, 0.0))
    cands.append(d)

    # 3x4 block - 1
    s = []
    for q in range(3):
        for r in range(4):
            if not (q == 0 and r == 3):
                s.append((math.sqrt(3) * (q + r / 2.0), 1.5 * r, 0.0))
    cands.append(s)

    # 5 + 6 (two rows of 5/6 in axial, offset)
    m = []
    for q in range(5):
        m.append((math.sqrt(3) * (q + 0.5), 1.5, 0.0))
    for q in range(6):
        m.append((math.sqrt(3) * q, 0.0, 0.0))
    cands.append(m)

    # elongated two rows
    e = []
    for q in range(6):
        e.append((math.sqrt(3) * q, 0.0, 0.0))
    for q in range(5):
        e.append((math.sqrt(3) * (q + 0.5), 1.5, 0.0))
    cands.append(e)

    # rotated star (30 deg) for variety
    rs = []
    for k in range(7):
        a = math.radians(30 + 60 * k)
        rr = 0.0 if k == 0 else math.sqrt(3)
        rs.append((rr * math.cos(a), rr * math.sin(a), 30.0))
    cands.append(rs)
    return cands

CANDIDATES = _build()


def _exact_S(pts, beta, cx, cy):
    """Exact enclosing-hexagon side length for given outer rotation/center."""
    angles = beta + math.pi / 6.0 + np.arange(6) * (math.pi / 3.0)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    pc = pts - np.array([cx, cy])
    projs = pc @ normals.T
    D = float(np.max(np.abs(projs)))
    return (D + APOTHEM) / APOTHEM


def _smooth_S(pts, beta, cx, cy, temp):
    angles = beta + math.pi / 6.0 + np.arange(6) * (math.pi / 3.0)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    pc = pts - np.array([cx, cy])
    projs = pc @ normals.T
    vals = np.concatenate([projs.ravel(), (-projs).ravel()])
    m = float(np.max(vals))
    if m > 1e6:
        return 1e6
    sh = (vals - m) / temp
    sh = np.clip(sh, -40.0, 40.0)
    D = temp * (np.log(np.sum(np.exp(sh))) + m / temp)
    return (D + APOTHEM) / APOTHEM


def _wrap_beta(b):
    return (b % (math.pi / 6.0) + math.pi / 6.0) % (math.pi / 6.0)


def _solve_candidate(pts0):
    """Optimize positions, per-hex rotations and outer (cx,cy,beta)."""
    n = len(pts0)
    best_S = float('inf')
    best_x = None

    # ---- Stage 1: smooth surrogate annealing (anneal temperature down) ----
    temps = [0.30, 0.15, 0.08, 0.04, 0.02]
    for temp in temps:
        if time.time() > _TIME_DEADLINE:
            break
        starts = [np.array(pts0, dtype=float).copy()]
        if best_x is not None:
            starts.append(best_x.copy())
        rng = np.random.default_rng(12345)
        for _ in range(3):
            p = np.array(pts0, dtype=float).copy()
            p[:, 0] += rng.normal(0, 0.3, n)
            p[:, 1] += rng.normal(0, 0.3, n)
            p[:, 2] += rng.uniform(-20, 20, n)
            starts.append(p)

        for p in starts:
            if time.time() > _TIME_DEADLINE:
                break
            x0 = np.concatenate([p, [0.0, 0.0, 0.0]])
            L = len(x0)
            m = 3 * L
            H = np.zeros((m, m))
            for i in range(L):
                H[i, i] = 1e-6
                H[L + i, L + i] = 1e-6
                H[2 * L + i, 2 * L + i] = 1e-6

            def obj(x, temp=temp):
                beta = _wrap_beta(x[-1])
                cx, cy = x[-3], x[-2]
                return _smooth_S(x[:3 * n].reshape(n, 3), beta, cx, cy, temp)

            try:
                res = minimize(obj, x0, method='L-BFGS-B',
                               jac='2-point', hess='2-point', hessp=H,
                               options={'maxiter': 300, 'gtol': 1e-9,
                                        'ftol': 1e-12, 'maxls': 40})
            except Exception:
                continue
            S = float(obj(res.x))
            if S < best_S:
                best_S = S
                best_x = res.x.copy()

    if best_x is None:
        return float('inf'), None

    # ---- Stage 2: exact Nelder-Mead polish (annealed restarts) ----
    for shrink in [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]:
        if time.time() > _TIME_DEADLINE:
            break
        base = best_x.copy()
        starts = [base]
        if shrink < 1.0:
            p = base[:3 * n].reshape(n, 3).copy()
            p[:, 0] *= shrink
            p[:, 1] *= shrink
            starts.append(np.concatenate([p.ravel(), base[-3:]]))

        for p in starts:
            if time.time() > _TIME_DEADLINE:
                break
            x0 = p.copy()

            def obj(x):
                beta = _wrap_beta(x[-1])
                cx, cy = x[-3], x[-2]
                return _exact_S(x[:3 * n].reshape(n, 3), beta, cx, cy)

            try:
                res = minimize(obj, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-13,
                                        'fatol': 1e-13})
            except Exception:
                continue
            S = float(obj(res.x))
            if S < best_S:
                best_S = S
                best_x = res.x.copy()

    # ---- Stage 3: ultra-precise final polish ----
    if time.time() < _TIME_DEADLINE:
        x0 = best_x.copy()

        def obj(x):
            beta = _wrap_beta(x[-1])
            cx, cy = x[-3], x[-2]
            return _exact_S(x[:3 * n].reshape(n, 3), beta, cx, cy)

        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-15,
                                    'fatol': 1e-15})
            S = float(obj(res.x))
            if S < best_S:
                best_S = S
                best_x = res.x.copy()
        except Exception:
            pass

    return best_S, best_x


def _valid(inner, center, side, angle_deg):
    """Check every inner hexagon lies inside the outer hexagon."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    D = APOTHEM * (side - 1.0)
    cx, cy = center
    for (ix, iy, ang) in inner:
        dx, dy = ix - cx, iy - cy
        b = math.radians(ang)
        cb, sb = math.cos(b), math.sin(b)
        for k in range(6):
            va = b + k * math.pi / 3.0
            vx, vy = math.cos(va), math.sin(va)
            lx = ca * (dx + cb * vx) + sa * (dy + sb * vy)
            ly = -sa * (dx + cb * vx) + ca * (dy + sb * vy)
            for j in range(6):
                fa = a + j * math.pi / 3.0
                nx, ny = math.cos(fa), math.sin(fa)
                if lx * nx + ly * ny > D + 1e-7:
                    return False
    return True


def construct():
    if time.time() > _TIME_DEADLINE:
        pts = _star()
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in pts]
        return inner, [cx, cy], 1.0, 0.0

    best_S = float('inf')
    best_x = None

    # compactness heuristic to order candidates
    def _compact(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        return max(math.hypot(p[0] - cx, p[1] - cy) for p in pts)

    order = sorted(range(len(CANDIDATES)), key=lambda i: _compact(CANDIDATES[i]))

    for idx in order:
        if time.time() > _TIME_DEADLINE:
            break
        S, x = _solve_candidate(CANDIDATES[idx])
        if x is not None and S < best_S:
            best_S = S
            best_x = x

    if best_x is None:
        # fallback: simple star with computed side
        pts = np.array(_star(), dtype=float)
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        S = _exact_S(pts, 0.0, cx, cy)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in pts]
        return inner, [cx, cy], float(S), 0.0

    n = len(best_x) // 3 - 1
    pts = best_x[:3 * n].reshape(n, 3)
    cx, cy, beta = float(best_x[-3]), float(best_x[-2]), _wrap_beta(float(best_x[-1]))

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in pts]
    angle_deg = math.degrees(beta)
    side = _exact_S(pts, beta, cx, cy)

    # safety: nudge side slightly up and verify containment
    for pad in [0.0, 1e-9, 1e-8, 1e-7, 1e-6]:
        s = side + pad
        if _valid(inner, [cx, cy], s, angle_deg):
            side = s
            break
    else:
        side = side + 1e-5

    return inner, [cx, cy], float(side), float(angle_deg)
