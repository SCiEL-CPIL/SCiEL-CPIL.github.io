# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 580.0  # safety margin before 600s/630s kill


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _separation(h1, h2):
    """Return min gap between two unit hexagons (>=0 if disjoint)."""
    v1 = _hex_vertices(*h1)
    v2 = _hex_vertices(*h2)
    best = 0.0
    for i in range(6):
        p = v1[i]
        q = v1[(i + 1) % 6]
        e = q - p
        nrm = np.array([-e[1], e[0]])
        nrm /= np.linalg.norm(nrm)
        # outward normal: away from hexagon center
        c = np.array([h1[0], h1[1]])
        if np.dot(nrm, p - c) < 0:
            nrm = -nrm
        proj1 = v1 @ nrm
        proj2 = v2 @ nrm
        gap = (proj1.min() - proj2.max()) if nrm[0] >= 0 else (proj2.min() - proj1.max())
        # compute both and take the one representing separation
        g1 = proj1.min() - proj2.max()
        g2 = proj2.min() - proj1.max()
        gap = max(g1, g2)
        if gap > best:
            best = gap
    return best


def _inner_outside_violation(inner, outer):
    """Sum of how far inner vertices lie outside the outer hexagon (0 if all inside)."""
    ov = _hex_vertices(*outer)
    n = len(ov)
    viol = 0.0
    for h in inner:
        v = _hex_vertices(*h)
        for p in v:
            for i in range(n):
                p1 = ov[i]
                p2 = ov[(i + 1) % n]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < 0:
                    viol += -cross
    return viol


def _outer_side_for(inner):
    """Min outer side (centered at origin, angle 0) such that all inner fit inside."""
    lo, hi = 3.0, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        outer = np.array([0.0, 0.0, 0.0, mid])
        if _inner_outside_violation(inner, outer) < 1e-9:
            hi = mid
        else:
            lo = mid
    return hi


def _objective(x, n=11):
    inner = x.reshape(n, 3)
    # penalty for overlaps
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sep = _separation(inner[i], inner[j])
            if sep < 1e-6:
                pen += (1e-6 - sep) ** 2 * 100.0
    side = _outer_side_for(inner)
    return side + pen


def hexagon_packing_11():
    n = 11
    # Good initial grid packing (seed)
    inner0 = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.165, 0],
        [1.25, 2.165, 0],
        [-1.25, -2.165, 0],
        [1.25, -2.165, 0],
        [-3.75, 2.165, 0],
        [3.75, 2.165, 0],
        [-3.75, -2.165, 0],
        [3.75, -2.165, 0],
    ])

    best_x = inner0.copy()
    best_side = _outer_side_for(inner0)

    # Try a few local optimizations with different starts
    starts = [inner0]
    # jittered starts
    rng = np.random.default_rng(0)
    for _ in range(3):
        starts.append(inner0 + rng.normal(0, 0.3, inner0.shape))

    for s0 in starts:
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
        res = minimize(_objective, s0.flatten(), method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-4, 'fatol': 1e-5})
        x = res.x.reshape(n, 3)
        # check validity of separation
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                if _separation(x[i], x[j]) < 1e-6:
                    ok = False
        if ok:
            side = _outer_side_for(x)
            if side < best_side:
                best_side = side
                best_x = x.copy()

    inner = best_x
    side = best_side
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(side)


# EVOLVE-BLOCK-END
