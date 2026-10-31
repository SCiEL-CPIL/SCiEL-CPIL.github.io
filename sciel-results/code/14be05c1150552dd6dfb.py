import numpy as np
import time
from math import sqrt, cos, sin, pi

N = 11
DEADLINE = 120.0

# Precompute unit hexagon vertices (pointy-top, centered at origin, side 1)
# vertices at angles 30,90,150,210,270,330 degrees
_UNIT_VERTS = np.array([
    [cos(pi/6), sin(pi/6)],
    [cos(pi/2), sin(pi/2)],
    [cos(5*pi/6), sin(5*pi/6)],
    [cos(7*pi/6), sin(7*pi/6)],
    [cos(3*pi/2), sin(3*pi/2)],
    [cos(11*pi/6), sin(11*pi/6)],
])

def _hex_verts(cx, cy, ang):
    c, s = cos(ang), sin(ang)
    R = np.array([[c, -s], [s, c]])
    return _UNIT_VERTS @ R.T + np.array([cx, cy])

def _poly_dist_sq(A, B):
    """Squared minimum distance between convex polygons A,B (n x 2).
    Returns (dist_sq, overlap_depth) where overlap_depth>0 if intersecting."""
    # Separating axis: edge normals of both
    n = len(A)
    best = 1e18
    overlap = 0.0
    for P, Q in ((A, B), (B, A)):
        for i in range(n):
            p0 = P[i]; p1 = P[(i+1) % n]
            e = p1 - p0
            L = np.hypot(e[0], e[1])
            if L < 1e-12:
                continue
            nrm = np.array([-e[1]/L, e[0]/L])
            projA = A @ nrm
            projB = B @ nrm
            minA, maxA = projA.min(), projA.max()
            minB, maxB = projB.min(), projB.max()
            gap = min(maxB, maxA) - max(minB, minA)
            if gap > 0:
                # separated along this axis
                return 0.0, 0.0
            else:
                d = -gap  # penetration depth along this axis
                if d > overlap:
                    overlap = d
    # No separating axis -> polygons intersect
    return 0.0, overlap

def _all_pairs_dist(A_verts, B_verts):
    """For a packing, compute pairwise min separation and containment margins.
    A_verts: list of (n x 2) arrays for each inner hex.
    Returns: (min_gap, min_margin) where
      min_gap = min over pairs of (min_dist - 0) [positive if disjoint]
      min_margin = min over all (outer_dist - 0) [positive if inside]
    We use penetration depth as negative gap."""
    nhex = len(A_verts)
    min_gap = 1e18
    for i in range(nhex):
        for j in range(i+1, nhex):
            dsq, ov = _poly_dist_sq(A_verts[i], A_verts[j])
            if ov > 0:
                gap = -ov
            else:
                gap = sqrt(dsq)
            if gap < min_gap:
                min_gap = gap
    return min_gap

def _outer_dist(hex_verts, ocx, ocy, oside, oang):
    """Minimum distance from hex_verts to outer hexagon boundary (positive if inside).
    We compute max penetration of hex_verts into the region outside the outer hex.
    Returns margin: positive means fully inside, negative means some vertex outside."""
    # Outer hexagon vertices
    ov = []
    for k in range(6):
        a = oang + k * pi / 3
        ov.append([ocx + oside * cos(a), ocy + oside * sin(a)])
    ov = np.array(ov)
    # For each inner vertex, check if inside outer hex; compute distance to boundary
    # Use half-plane test: outer hex edges, inward normal
    worst = 1e18
    for k in range(6):
        p0 = ov[k]; p1 = ov[(k+1) % 6]
        e = p1 - p0
        L = np.hypot(e[0], e[1])
        nrm = np.array([-e[1]/L, e[0]/L])
        # ensure inward: point toward center
        to_center = np.array([ocx, ocy]) - p0
        if np.dot(nrm, to_center) < 0:
            nrm = -nrm
        # signed distance of each inner vertex: positive = inside (toward center side)
        d = (hex_verts - p0) @ nrm
        # margin = min d ; if any < 0, vertex is outside
        m = d.min()
        if m < worst:
            worst = m
    return worst

def _objective(params, oside):
    """params: (N x 3) [x, y, ang]. Returns (gap, margin)."""
    verts = []
    for k in range(N):
        verts.append(_hex_verts(params[k,0], params[k,1], params[k,2]))
    gap = _all_pairs_dist(verts, verts)
    margin = 1e18
    for v in verts:
        m = _outer_dist(v, 0.0, 0.0, oside, 0.0)
        if m < margin:
            margin = m
    return gap, margin

def _oside_needed(params):
    """Find minimal oside such that margin >= 0 via bisection."""
    # upper bound
    lo = 1.0
    hi = 200.0
    # ensure hi is large enough
    for _ in range(20):
        _, m = _objective(params, hi)
        if m >= 0:
            break
        hi *= 2
    for _ in range(40):
        mid = (lo + hi) / 2
        _, m = _objective(params, mid)
        if m >= 0:
            hi = mid
        else:
            lo = mid
    return hi

def _energy(params, oside, wgap, wmargin):
    """Smooth energy: penalize overlap and outside."""
    gap, margin = _objective(params, oside)
    # soft penalty
    e = 0.0
    if gap < 0:
        e += wgap * gap * gap
    if margin < 0:
        e += wmargin * margin * margin
    return e

def _grad_energy(params, oside, wgap, wmargin, step=1e-4):
    g = np.zeros_like(params)
    base = _energy(params, oside, wgap, wmargin)
    for i in range(params.shape[0]):
        for j in range(3):
            p = params.copy()
            p[i,j] += step
            ep = _energy(p, oside, wgap, wmargin)
            p[i,j] -= 2*step
            em = _energy(p, oside, wgap, wmargin)
            g[i,j] = (ep - em) / (2*step)
    return g

def _initial_params():
    # hexagonal lattice arrangement
    params = np.zeros((N, 3))
    idx = 0
    # rows
    for r in range(N):
        for c in range(N):
            if idx >= N:
                break
            x = c * 1.74 + (r % 2) * 0.87
            y = r * 1.5
            params[idx] = [x, y, 0.0]
            idx += 1
        if idx >= N:
            break
    return params

def construct():
    t0 = time.time()
    best_params = _initial_params()
    best_oside = _oside_needed(best_params)
    best_val = best_oside

    # Simulated annealing on the joint space, minimizing oside via smooth energy
    # Use a fixed reference oside that we shrink; better: optimize params to maximize
    # (gap, margin) jointly, then oside = needed.
    # We'll minimize a combined smooth objective:
    #   F = -min(gap, margin) approximated by soft-min, plus oside cost.
    # Simpler: minimize energy at a target oside, and adapt target.
    rng = np.random.default_rng(12345)
    T = 5.0
    T_end = 1e-6
    cooling = 0.99985
    params = best_params.copy()
    # current target oside
    target = best_oside
    wgap = 1.0
    wmargin = 1.0
    iters = 0
    while time.time() - t0 < DEADLINE - 5.0:
        iters += 1
        # perturb
        step = 0.5 * T
        new_params = params + rng.normal(0, step, params.shape)
        # keep angles in [0, pi)
        new_params[:,2] = np.mod(new_params[:,2], pi)
        e_old = _energy(params, target, wgap, wmargin)
        e_new = _energy(new_params, target, wgap, wmargin)
        if e_new <= e_old or rng.random() < np.exp((e_old - e_new) / max(T, 1e-9)):
            params = new_params
            # periodically recompute needed oside and update best
            if iters % 50 == 0:
                needed = _oside_needed(params)
                if needed < best_val:
                    best_val = needed
                    best_params = params.copy()
                    target = needed
        T *= cooling
        if T < T_end:
            break
    # polish: local gradient descent on energy at target=best_val
    target = best_val
    p = best_params.copy()
    for _ in range(200):
        if time.time() - t0 > DEADLINE - 3.0:
            break
        g = _grad_energy(p, target, 1.0, 1.0)
        # gradient step with backtracking
        lr = 0.05
        e0 = _energy(p, target, 1.0, 1.0)
        accepted = False
        for _ in range(10):
            p2 = p - lr * g
            p2[:,2] = np.mod(p2[:,2], pi)
            e2 = _energy(p2, target, 1.0, 1.0)
            if e2 < e0:
                p = p2
                e0 = e2
                accepted = True
                break
            lr *= 0.5
        if not accepted:
            break
    needed = _oside_needed(p)
    if needed < best_val:
        best_val = needed
        best_params = p.copy()

    # Build inner list
    inner = [[float(best_params[k,0]), float(best_params[k,1]),
              float(np.degrees(best_params[k,2]))] for k in range(N)]
    return inner, (0.0, 0.0), float(best_val), 0.0
