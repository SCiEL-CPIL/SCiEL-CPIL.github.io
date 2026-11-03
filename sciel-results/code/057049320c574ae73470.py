import time
import numpy as np
from scipy.optimize import minimize

# ---- time budget (wall-clock) ----
_T0 = time.time()
_LIMIT = 130.0          # hard wall-clock safety margin
_SOFT = 115.0           # soft: stop starting new multi-starts
_EPS = 1e-6

def _time_left():
    return _LIMIT - (time.time() - _T0)

# ---- geometry helpers ----
def _verts(cx, cy, R, rot):
    a = np.deg2rad(rot)
    ang = a + np.arange(6) * np.pi / 3.0
    return np.column_stack((cx + R * np.cos(ang), cy + R * np.sin(ang)))

def _separated(A, B, eps=_EPS):
    """True if two convex polygons are disjoint (separating axis)."""
    for P in (A, B):
        for i in range(6):
            x1, y1 = P[i]
            x2, y2 = P[(i + 1) % 6]
            ax, ay = y1 - y2, x2 - x1          # edge normal
            n = np.hypot(ax, ay)
            if n < 1e-12:
                continue
            ax, ay = ax / n, ay / n
            minA = np.min(A[:, 0] * ax + A[:, 1] * ay)
            maxA = np.max(A[:, 0] * ax + A[:, 1] * ay)
            minB = np.min(B[:, 0] * ax + B[:, 1] * ay)
            maxB = np.max(B[:, 0] * ax + B[:, 1] * ay)
            if maxA + eps < minB or maxB + eps < minA:
                return True
    return False

def _fits(inner, outer, eps=_EPS):
    """True if inner hexagon is inside outer hexagon (support containment)."""
    for i in range(6):
        x1, y1 = outer[i]
        x2, y2 = outer[(i + 1) % 6]
        ax, ay = y1 - y2, x2 - x1
        n = np.hypot(ax, ay)
        if n < 1e-12:
            continue
        ax, ay = ax / n, ay / n
        if np.max(inner[:, 0] * ax + inner[:, 1] * ay) + eps > np.max(outer[:, 0] * ax + outer[:, 1] * ay):
            return False
    return True

def _feasible(pos, cx, cy, R, rot):
    outer = _verts(cx, cy, R, rot)
    inners = [_verts(pos[k, 0], pos[k, 1], pos[k, 2]) for k in range(N)]
    for k in range(N):
        if not _fits(inners[k], outer):
            return False
    for k in range(N):
        for m in range(k + 1, N):
            if not _separated(inners[k], inners[m]):
                return False
    return True

def _bounds_hex(pos, rot=0.0):
    """Smallest (centered-at-mean, rotation=rot) hexagon containing all inners."""
    pts = np.vstack([_verts(pos[k, 0], pos[k, 1], pos[k, 2]) for k in range(N)])
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    a = np.deg2rad(rot)
    best = 1e9
    for k in range(6):
        ax, ay = np.cos(a + (k + 0.5) * np.pi / 3.0), np.sin(a + (k + 0.5) * np.pi / 3.0)
        best = max(best, np.max((pts - [cx, cy]) @ [ax, ay]))
    return cx, cy, best

# ---- objective / constraints (reduced variables: x,y,angle per hex + cx,cy,R) ----
def _build(pos, cx, cy, R, rot):
    outer = _verts(cx, cy, R, rot)
    inners = [_verts(pos[k, 0], pos[k, 1], pos[k, 2]) for k in range(N)]
    return inners, outer

def _objective(v, rot):
    pos = v[:3 * N].reshape(N, 3)
    cx, cy, R = v[3 * N], v[3 * N + 1], v[3 * N + 2]
    return R

def _constraints(rot):
    cons = []
    for k in range(N):
        def c_in(v, k=k, rot=rot):
            pos = v[:3 * N].reshape(N, 3)
            inners, outer = _build(pos, v[3 * N], v[3 * N + 1], v[3 * N + 2], rot)
            ok = _fits(inners[k], outer)
            return 1.0 if ok else 0.0
        cons.append({'type': 'ineq', 'fun': c_in})
    for k in range(N):
        for m in range(k + 1, N):
            def c_ov(v, k=k, m=m, rot=rot):
                pos = v[:3 * N].reshape(N, 3)
                inners, _ = _build(pos, v[3 * N], v[3 * N + 1], v[3 * N + 2], rot)
                return 1.0 if _separated(inners[k], inners[m]) else 0.0
            cons.append({'type': 'ineq', 'fun': c_ov})
    return cons

def _optimize(x0, rot, iters):
    pos0 = x0[:3 * N].reshape(N, 3)
    cx0, cy0, R0 = x0[3 * N], x0[3 * N + 1], x0[3 * N + 2]
    cons = _constraints(rot)
    res = minimize(_objective, x0, args=(rot,), method='SLSQP',
                   constraints=cons, bounds=None,
                   options={'maxiter': iters, 'ftol': 1e-8, 'disp': False})
    if res.x is None:
        return None
    pos = res.x[:3 * N].reshape(N, 3)
    cx, cy, R = res.x[3 * N], res.x[3 * N + 1], res.x[3 * N + 2]
    if R <= 0 or not np.all(np.isfinite(res.x)):
        return None
    if not _feasible(pos, cx, cy, R, rot):
        return None
    return pos, cx, cy, R

def _ring(r):
    """Deterministic ring of 12 hexagon centers, radius r."""
    ang = np.arange(N) * 2.0 * np.pi / N
    return np.column_stack((r * np.cos(ang), r * np.sin(ang), np.zeros(N)))

def _start(k):
    r = 2.0 + 0.15 * (k % 8)
    pos = _ring(r)
    if k % 3 == 1:
        pos[:, 2] = np.deg2rad(30.0)
    if k % 3 == 2:
        pos[:, 2] = np.deg2rad(15.0)
    # small deterministic jitter to break symmetry
    pos = pos + 0.03 * np.array([[0.5, -0.5, 0.0], [0.0, 0.5, 0.0], [-0.5, 0.0, 0.0]])
    cx, cy, R = _bounds_hex(pos)
    R *= 1.02
    return np.concatenate([pos.ravel(), [cx, cy, R]])

N = 12

# EVOLVE-BLOCK-START
def construct():
    best = None
    bestR = 1e18
    iters = 400
    k = 0
    while _time_left() > 12.0:
        if time.time() - _T0 > _SOFT:
            break
        x0 = _start(k)
        res = _optimize(x0, 0.0, iters)
        if res is not None:
            pos, cx, cy, R = res
            if R < bestR:
                best = (pos, cx, cy, R)
                bestR = R
        k += 1

    if best is None:
        # deterministic analytical fallback: ring of 12, bounding hexagon
        pos = _ring(2.0)
        cx, cy, R = _bounds_hex(pos)
        R *= 1.05
        return [[float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])] for i in range(N)], \
               (float(cx), float(cy)), float(R), 0.0

    pos, cx, cy, R = best
    # tighten outer side by shrinking R until infeasible
    lo, hi = 0.0, R
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _feasible(pos, cx, cy, mid, 0.0):
            lo = mid
        else:
            hi = mid
    R = lo * 1.0005

    return [[float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])] for i in range(N)], \
           (float(cx), float(cy)), float(R), 0.0
# EVOLVE-BLOCK-END
