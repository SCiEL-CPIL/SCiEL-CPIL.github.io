"""Heilbronn: maximize min triangle area over 11 points in a unit equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
# Triangle vertices (unit area): A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
H = np.sqrt(3.0) / 2.0
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])
DEADLINE = 112.0  # safety margin under 120s limit


def _area(p, q, r):
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))


def _min_area(pts):
    m = 0.0
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _area(pts[i], pts[j], pts[k])
                if a > m:
                    m = a
    return m


def _soft_min(pts, beta):
    """Smooth surrogate for the minimum triangle area (lower beta -> closer to true min)."""
    vals = []
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                vals.append(_area(pts[i], pts[j], pts[k]))
    vals = np.array(vals)
    vmax = vals.max()
    s = np.exp((vals - vmax) * beta)
    return vmax - np.log(s.sum()) / beta


def _bary_to_cart(u):
    """u is (N,2); map (u,v) barycentric-ish coords to cartesian inside triangle."""
    u = np.asarray(u, dtype=float).reshape(N, 2)
    return np.column_stack((u[:, 0] + 0.5 * u[:, 1], H * u[:, 1]))


def _cart_to_bary(pts):
    pts = np.asarray(pts, dtype=float).reshape(N, 2)
    u = pts[:, 0] - 0.5 * pts[:, 1] / H
    v = pts[:, 1] / H
    return np.column_stack((u, v))


def _in_triangle(pts, margin=0.0):
    """Check all points inside triangle with given margin."""
    for p in pts:
        # barycentric test
        v0 = B - A
        v1 = C - A
        v2 = p - A
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d02 = np.dot(v0, v2)
        d11 = np.dot(v1, v1)
        d12 = np.dot(v1, v2)
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return False
        v = (d11 * d02 - d01 * d12) / denom
        w = (d00 * d12 - d01 * d02) / denom
        if v < -margin or w < -margin or v + w > 1 + margin:
            return False
    return True


def _neg_soft_min(u, beta):
    pts = _bary_to_cart(u)
    return -_soft_min(pts, beta)


def _neg_soft_min_grad(u, beta):
    pts = _bary_to_cart(u)
    n = len(pts)
    grads = np.zeros_like(pts)
    vals = []
    idxs = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _area(pts[i], pts[j], pts[k])
                vals.append(a)
                idxs.append((i, j, k))
    vals = np.array(vals)
    vmax = vals.max()
    s = np.exp((vals - vmax) * beta)
    ss = s.sum()
    sm = vmax - np.log(ss) / beta
    # d sm / d area_i = -(s_i / ss)
    dsm = -(s / ss)
    for m, (i, j, k) in enumerate(idxs):
        d = dsm[m]
        p, q, r = pts[i], pts[j], pts[k]
        # cross = (q-p)x(r-p)
        cross = (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])
        if cross >= 0:
            sgn = 1.0
        else:
            sgn = -1.0
        # d area / d p = -0.5 * sgn * ( (r[1]-q[1]), -(r[0]-q[0]) )  (using cross of (q-p),(r-p))
        # derivative of cross w.r.t p: cross = (q0-p0)(r1-p1) - (r0-p0)(q1-p1)
        # d/dp0 = -(r1-p1) - (-(q1-p1)) = -(r1-p1)+(q1-p1)
        # d/dp1 = -(q0-p0) + (r0-p0)
        dp = np.array([-(r[1] - p[1]) + (q[1] - p[1]), -(q[0] - p[0]) + (r[0] - p[0])])
        # d/dq0 = (r1-p1); d/dq1 = -(r0-p0)
        dq = np.array([r[1] - p[1], -(r[0] - p[0])])
        # d/dr0 = -(q1-p1); d/dr1 = (q0-p0)
        dr = np.array([-(q[1] - p[1]), q[0] - p[0]])
        grads[i] += 0.5 * sgn * d * dp
        grads[j] += 0.5 * sgn * d * dq
        grads[k] += 0.5 * sgn * d * dr
    # map cartesian grad -> barycentric grad
    # cart_x = u + 0.5 v ; cart_y = H v
    # d cart_x / d u = 1 ; d cart_x / d v = 0.5
    # d cart_y / d u = 0 ; d cart_y / d v = H
    gu = grads[:, 0]
    gv = grads[:, 0] * 0.5 + grads[:, 1] / H
    return -np.column_stack((gu, gv))


def _solve_one(u0, beta, deadline):
    u0 = np.asarray(u0, dtype=float).reshape(N, 2)
    res = minimize(
        _neg_soft_min,
        u0,
        jac=lambda x: _neg_soft_min_grad(x, beta),
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * (2 * N),
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 400},
    )
    return res.x


def construct():
    t0 = time.time()
    rng = np.random.default_rng(7)
    best_u = None
    best_val = -1.0

    def _update(u):
        nonlocal best_u, best_val
        pts = _bary_to_cart(u)
        if not _in_triangle(pts):
            return
        v = _min_area(pts)
        if v > best_val:
            best_val = v
            best_u = u.copy()

    # ---- Phase 1: coarse soft-min optimization from several starts ----
    starts = []
    # grid-like starts
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i + j + k <= 2:
                    u = np.array([i / 4.0, j / 4.0], dtype=float)
                    starts.append(u)
    # random starts
    for _ in range(8):
        u = rng.random(N, 2) * 0.8 + 0.1
        starts.append(u)

    # build initial point sets from starts by distributing points
    # We'll just use random-ish full sets but bias toward grid
    initial_sets = []
    for s in starts:
        # create N points around the start
        pts = np.tile(s, (N, 1)) + (rng.random(N, 2) - 0.5) * 0.3
        pts = np.clip(pts, 0.0, 1.0)
        initial_sets.append(pts)
    # also a few purely random sets
    for _ in range(4):
        initial_sets.append(rng.random(N, 2) * 0.9 + 0.05)

    beta_coarse = 30.0
    for init in initial_sets:
        if time.time() > DEADLINE - 25:
            break
        try:
            u = _solve_one(init, beta_coarse, DEADLINE)
            _update(u)
        except Exception:
            continue

    # ---- Phase 2: refine with larger beta (sharper min) ----
    if best_u is not None and time.time() < DEADLINE - 15:
        for beta in [60.0, 120.0]:
            if time.time() > DEADLINE - 10:
                break
            try:
                u = _solve_one(best_u, beta, DEADLINE)
                _update(u)
            except Exception:
                continue

    # ---- Phase 3: high-precision polish on the best point set ----
    if best_u is not None and time.time() < DEADLINE - 4:
        # local gradient-free polish: small perturbations of the worst triangle
        pts = _bary_to_cart(best_u)
        cur = _min_area(pts)
        for it in range(200):
            if time.time() > DEADLINE - 3:
                break
            # find the triangle with min area
            worst = None
            wval = 1e9
            for i in range(N):
                for j in range(i + 1, N):
                    for k in range(j + 1, N):
                        a = _area(pts[i], pts[j], pts[k])
                        if a < wval:
                            wval = a
                            worst = (i, j, k)
            if worst is None:
                break
            i, j, k = worst
            # try to move the three points slightly apart to increase area
            improved = False
            for step in [0.005, 0.002, 0.001, 0.0005]:
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            np_ = pts.copy()
                            # move i and j apart
                            np_[i, 0] += di * step
                            np_[j, 0] += dj * step
                            np_[k, 1] += dk * step
                            # keep inside triangle
                            if not _in_triangle(np_):
                                continue
                            v = _min_area(np_)
                            if v > cur + 1e-12:
                                pts = np_
                                cur = v
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        _update(_cart_to_bary(pts))

    if best_u is None:
        # fallback: return the seed-style random points
        points = []
        import random
        r = random.Random(7)
        while len(points) < N:
            u, v = r.random(), r.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, H * v])
        return np.array(points)

    return _bary_to_cart(best_u)
