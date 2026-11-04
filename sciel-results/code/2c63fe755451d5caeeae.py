"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0
_t0 = time.time()

# Six outer normals (unit hexagon, flat-top)
COS60 = 0.5
SIN60 = math.sqrt(3.0) / 2.0
NORMS = np.array([
    [1.0, 0.0],
    [COS60, SIN60],
    [-COS60, SIN60],
    [-1.0, 0.0],
    [-COS60, -SIN60],
    [COS60, -SIN60],
])
# 7 vertices of a unit hexagon (center + 6 corners) at angle 0
V7 = np.vstack([np.zeros(2), np.column_stack([np.cos(np.arange(6) * np.pi / 3.0),
                                                np.sin(np.arange(6) * np.pi / 3.0)])])


def _elapsed():
    return time.time() - _t0


def _feasible_side(x, c, L):
    """Return max over all (i,j,k) of dot(V7[j], n[k]) + dot(c, n[k]) - dot(pos[i], n[k]).
    Feasible iff this max <= L."""
    # pos: (12,2), c: (2,), V7: (7,2), NORMS: (6,2)
    # term[i,j,k] = dot(V7[j], n[k]) + dot(c, n[k]) - dot(pos[i], n[k])
    # = vdot[j,k] + cdot[k] - pdot[i,k]
    vdot = V7 @ NORMS.T          # (7,6)
    cdot = c @ NORMS.T           # (6,)
    pdot = x @ NORMS.T           # (12,6)
    # max over i for each (j,k): - min_i pdot[i,k] + vdot[j,k] + cdot[k]
    # min over i of pdot[i,k] -> -min = max of -pdot
    max_term = vdot + cdot[None, :] - pdot.min(axis=0)[None, :]
    return float(max_term.max())


def _objective(x, c):
    # c is fixed (0,0) for the main optimization
    return _feasible_side(x, c, 0.0)


def _hexagonal_start():
    """1 center + 6 ring + 5 of a second ring (skip 1 to keep 12)."""
    d = 2.0 * SIN60  # center-to-center for touching = 2*apothem = sqrt(3)
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = k * np.pi / 3.0
        pts.append((d * math.cos(a), d * math.sin(a)))
    # second ring: 12 positions at distance 2d, angles (k+0.5)*pi/3
    for k in range(5):
        a = (k + 0.5) * np.pi / 3.0
        pts.append((2.0 * d * math.cos(a), 2.0 * d * math.sin(a)))
    x = np.array(pts, dtype=float)
    theta = np.zeros(N)
    return np.concatenate([x.ravel(), theta])


def _random_start(scale):
    rng = np.random.default_rng()
    x = rng.uniform(-scale, scale, size=(N, 2))
    theta = rng.uniform(0.0, np.pi / 3.0, size=N)
    return np.concatenate([x.ravel(), theta])


def _unroll(v):
    x = v[:2 * N].reshape(N, 2)
    theta = v[2 * N:]
    return x, theta


def _solve_once(v0, c, maxit, deadline):
    """Run L-BFGS-B; return (v, L) or None."""
    x, theta = _unroll(v0)
    L0 = _feasible_side(x, c, 0.0)

    def obj(v):
        x, _ = _unroll(v)
        return _feasible_side(x, c, 0.0)

    try:
        res = minimize(obj, v0, method='L-BFGS-B',
                       options={'maxiter': maxit, 'maxls': 40,
                                'ftol': 1e-12, 'gtol': 1e-9})
    except Exception:
        return None
    if _elapsed() > deadline:
        return None
    v = res.x
    L = obj(v)
    return v, L


def construct():
    deadline = DEADLINE - 10.0  # safety margin
    c = np.array([0.0, 0.0])

    # Phase 1: multi-start local optimization
    starts = [_hexagonal_start()]
    rng = np.random.default_rng(0)
    for _ in range(8):
        starts.append(_random_start(3.0))

    best_v = None
    best_L = float('inf')
    for s in starts:
        if _elapsed() > deadline - 5.0:
            break
        out = _solve_once(s, c, maxit=300, deadline=deadline)
        if out is None:
            continue
        v, L = out
        if L < best_L:
            best_L = L
            best_v = v

    if best_v is None:
        # fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Phase 2: polish with tighter tolerances and more iterations
    if _elapsed() < deadline - 10.0:
        def obj(v):
            x, _ = _unroll(v)
            return _feasible_side(x, c, 0.0)
        try:
            res = minimize(obj, best_v, method='L-BFGS-B',
                           options={'maxiter': 2000, 'maxls': 50,
                                    'ftol': 1e-14, 'gtol': 1e-11})
            if res.success or res.fun < best_L:
                best_v = res.x
                best_L = obj(best_v)
        except Exception:
            pass

    # Phase 3: try a tiny global nudge of the outer center (breaks symmetry)
    # We keep c=0 by symmetry, but allow a small shift search if time remains.
    if _elapsed() < deadline - 5.0:
        # small 1-D scan over c_x in [-0.2, 0.2] to see if shifting helps
        best_shift = (0.0, 0.0, best_L)
        for cx in np.linspace(-0.15, 0.15, 7):
            if _elapsed() > deadline - 3.0:
                break
            cc = np.array([cx, 0.0])
            x, theta = _unroll(best_v)
            Lc = _feasible_side(x, cc, 0.0)
            # also re-optimize inner given this c briefly
            def obj_c(v):
                xx, _ = _unroll(v)
                return _feasible_side(xx, cc, 0.0)
            try:
                res = minimize(obj_c, best_v, method='L-BFGS-B',
                               options={'maxiter': 200, 'maxls': 30,
                                        'ftol': 1e-12, 'gtol': 1e-10})
                Lc = obj_c(res.x)
                if Lc < best_shift[2] - 1e-9:
                    best_shift = (cx, 0.0, Lc)
                    best_v = res.x
                    best_L = Lc
            except Exception:
                pass

    x, theta = _unroll(best_v)
    inner = []
    for i in range(N):
        ang_deg = math.degrees(theta[i] % (2 * math.pi))
        inner.append([float(x[i, 0]), float(x[i, 1]), float(ang_deg)])

    # outer center (0,0), side = best_L, rotation 0
    return inner, (0.0, 0.0), float(best_L), 0.0


if __name__ == '__main__':
    inner, center, side, angle = construct()
    print("side =", side)
    for r in inner:
        print(r)
