"""Circle packing in perimeter-4 rectangle: convex SOC formulation + Delaunay contact graph.

Strategy:
  - Non-overlap ||c_i - c_j|| >= r_i + r_j is a second-order cone constraint, jointly
    convex in (c, r).  Boundary r_i <= min(x_i, w-x_i, y_i, h-y_i) is linear.
  - For fixed aspect ratio the problem is convex -> SLSQP (interior-point) finds the
    global optimum of sum(r).
  - Aspect ratio optimized via 1-D golden-section (unimodal in 1-D).
  - Delaunay triangulation gives a sparse contact graph (~60 edges) instead of 210 pairs.
  - Nelder-Mead polish on the contact graph.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
DEADLINE = 150.0   # hard wall-clock kill by evaluator
SAFETY = 20.0      # return this much before the kill
LIMIT = DEADLINE - SAFETY


def _is_valid(arr, w, h, tol=1e-7):
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(rs <= 0):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d < rs[i] + rs[j] - tol:
                return False
    return True


def _perimeter(arr):
    x0, x1 = np.min(arr[:, 0] - arr[:, 2]), np.max(arr[:, 0] + arr[:, 2])
    y0, y1 = np.min(arr[:, 1] - arr[:, 2]), np.max(arr[:, 1] + arr[:, 2])
    return 2.0 * ((x1 - x0) + (y1 - y0)), (x1 - x0), (y1 - y0)


def _contact_edges(arr, tol=1e-4):
    """Edges of the Delaunay graph (sparse contact graph)."""
    pts = arr[:, :2]
    try:
        tri = Delaunay(pts)
        edges = set()
        for t in tri.simplices:
            edges.add(tuple(sorted((t[0], t[1]))))
            edges.add(tuple(sorted((t[1], t[2]))))
            edges.add(tuple(sorted((t[0], t[2]))))
        return sorted(edges)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _build_constraints(arr, w, h, tol=1e-4):
    """Return (ineq_funcs, ineq_bounds) for SLSQP.  f(x) >= 0 form."""
    edges = _contact_edges(arr, tol)
    funcs = []
    bounds = []
    for (i, j) in edges:
        funcs.append(lambda z, i=i, j=j:
                     np.hypot(z[2 * i] - z[2 * j], z[2 * i + 1] - z[2 * j + 1])
                     - z[2 * i + 2] - z[2 * j + 2])
        bounds.append(0.0)
    # boundary constraints: r_i <= x_i, w-x_i, y_i, h-y_i  ->  x_i - r_i >= 0 etc.
    for i in range(N):
        funcs.append(lambda z, i=i: z[2 * i] - z[2 * i + 2])
        bounds.append(0.0)
        funcs.append(lambda z, i=i: (w - z[2 * i]) - z[2 * i + 2])
        bounds.append(0.0)
        funcs.append(lambda z, i=i: z[2 * i + 1] - z[2 * i + 2])
        bounds.append(0.0)
        funcs.append(lambda z, i=i: (h - z[2 * i + 1]) - z[2 * i + 2])
        bounds.append(0.0)
    return funcs, bounds


def _slsqp_objective(z, w, h, edges):
    """Return (f, g):  f = -sum(r)  (minimize),  g_k = violation_k (<=0 form)."""
    r = z[2::3]
    f = -np.sum(r)
    g = []
    for (i, j) in edges:
        d = np.hypot(z[2 * i] - z[2 * j], z[2 * i + 1] - z[2 * j + 1])
        g.append(r[i] + r[j] - d)
    for i in range(N):
        g.append(r[i] - z[2 * i])
        g.append(r[i] - (w - z[2 * i]))
        g.append(r[i] - z[2 * i + 1])
        g.append(r[i] - (h - z[2 * i + 1]))
    return f, np.array(g)


def _slsqp_solve(arr, w, h, time_left, maxiter=400):
    """Solve the convex packing problem for fixed (w,h) with SLSQP."""
    edges = _contact_edges(arr)
    z0 = arr.reshape(-1).copy()
    bounds = []
    for i in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((1e-8, None))
    t0 = time.time()
    try:
        res = minimize(_slsqp_objective, z0, args=(w, h, edges), method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False})
    except Exception:
        return None, 0.0
    if time.time() - t0 > time_left:
        return None, 0.0
    if res.x is None:
        return None, 0.0
    sol = res.x.reshape(N, 3)
    if not _is_valid(sol, w, h):
        return None, 0.0
    return sol, float(np.sum(sol[:, 2]))


def _nm_polish(arr, w, h, time_left, maxiter=2000):
    """Nelder-Mead on the contact graph to polish positions (radii fixed)."""
    edges = _contact_edges(arr)
    r = arr[:, 2].copy()

    def obj(z):
        # penalty: maximize min margin over contact edges + boundaries
        m = np.inf
        for (i, j) in edges:
            d = np.hypot(z[2 * i] - z[2 * j], z[2 * i + 1] - z[2 * j + 1])
            m = min(m, d - r[i] - r[j])
        for i in range(N):
            m = min(m, z[2 * i] - r[i], w - z[2 * i] - r[i],
                    z[2 * i + 1] - r[i], h - z[2 * i + 1] - r[i])
        return -m

    z0 = arr[:, :2].reshape(-1).copy()
    t0 = time.time()
    try:
        res = minimize(obj, z0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-12})
    except Exception:
        return arr
    if time.time() - t0 > time_left:
        return arr
    sol = arr.copy()
    sol[:, :2] = res.x.reshape(N, 2)
    if _is_valid(sol, w, h):
        return sol
    return arr


def _full_expand(arr, w, h, max_iters=200):
    """Iteratively grow each radius to its constraint limit (coordinate ascent)."""
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            x, y, r = arr[i]
            rmax = min(x, w - x, y, h - y)
            for j in range(N):
                if j == i:
                    continue
                d = np.hypot(x - arr[j, 0], y - arr[j, 1])
                rmax = min(rmax, d - arr[j, 2])
            rmax = max(rmax, 0.0)
            if rmax > r + 1e-12:
                arr[i, 2] = rmax
                improved = True
        if not improved:
            break
    return arr


def _init_grid(w, h):
    side = int(np.ceil(np.sqrt(N)))
    arr = np.zeros((N, 3))
    c = 0
    for i in range(side):
        for j in range(side):
            if c >= N:
                break
            arr[c, 0] = (i + 0.5) * w / side
            arr[c, 1] = (j + 0.5) * h / side
            arr[c, 2] = min(w, h) / (2 * side) * 0.7
            c += 1
    return arr


def _init_hex(w, h):
    arr = np.zeros((N, 3))
    r_est = min(w, h) / 9.0
    idx = 0
    row = 0
    col = 0
    while idx < N and row < 20:
        x = (col + 0.5) * r_est * np.sqrt(3) + (r_est * np.sqrt(3) / 2 if row % 2 else 0)
        y = (row + 0.5) * r_est * 1.5 + r_est
        if x + r_est <= w and y + r_est <= h:
            arr[idx] = [x, y, r_est]
            idx += 1
            col += 1
        else:
            col = 0
            row += 1
    if idx < N:
        side = int(np.ceil(np.sqrt(N - idx)))
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                arr[idx, 0] = (i + 0.5) * w / side
                arr[idx, 1] = (j + 0.5) * h / side
                arr[idx, 2] = min(w, h) / (2 * side) * 0.7
                idx += 1
    return arr


def _solve_aspect(aspect, start_time, inits=None):
    """Solve the packing for a given aspect ratio; return (arr, sum_r, w, h)."""
    h = 2.0 / (aspect + 1.0)
    w = aspect * h
    best_arr, best_sum = None, -np.inf
    if inits is None:
        inits = [_init_grid(w, h), _init_hex(w, h)]
    for arr0 in inits:
        if time.time() - start_time > LIMIT:
            break
        if not _is_valid(arr0, w, h):
            for s in (0.8, 0.6, 0.4):
                a = arr0.copy(); a[:, 2] *= s
                if _is_valid(a, w, h):
                    arr0 = a; break
            else:
                continue
        # convex solve
        t_left = LIMIT - (time.time() - start_time)
        if t_left < 2.0:
            break
        sol, s = _slsqp_solve(arr0, w, h, t_left, maxiter=400)
        if sol is not None and s > best_sum:
            best_arr, best_sum = sol, s
        # NM polish on best found so far
        if best_arr is not None:
            t_left = LIMIT - (time.time() - start_time)
            if t_left > 3.0:
                polished = _nm_polish(best_arr, w, h, t_left, maxiter=1500)
                sp = float(np.sum(polished[:, 2]))
                if sp > best_sum:
                    best_arr, best_sum = polished, sp
    if best_arr is None:
        return None, -np.inf, w, h
    return best_arr, best_sum, w, h


def _golden_aspect(start_time, lo=0.4, hi=3.0, inits_fn=None):
    """1-D golden-section search over aspect ratio (unimodal in 1-D)."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _solve_aspect(c, start_time, inits_fn)
    fd = _solve_aspect(d, start_time, inits_fn)
    best = max((fc, c), (fd, d), key=lambda t: t[0][1])
    while (b - a) > 1e-3 and (time.time() - start_time) < LIMIT:
        if fc[1] > fd[1]:
            b = d; d = c; fd = fc
            c = b - phi * (b - a)
            fc = _solve_aspect(c, start_time, inits_fn)
        else:
            a = c; c = d; fc = fd
            d = a + phi * (b - a)
            fd = _solve_aspect(d, start_time, inits_fn)
        best = max(best, (fc, c), (fd, d), key=lambda t: t[0][1])
    return best[0], best[1]


def construct():
    start_time = time.time()
    random.seed(12345)
    np.random.seed(12345)

    best_arr, best_sum = None, -np.inf

    def _record(arr, s):
        nonlocal best_arr, best_sum
        if arr is not None and s > best_sum:
            best_arr, best_sum = arr.copy(), s

    # Phase 1: golden-section over aspect ratio (convex solve per aspect)
    arr, s, w, h = _golden_aspect(start_time, lo=0.5, hi=2.5)
    _record(arr, s)

    # Phase 2: refine around the best aspect with a few local aspects
    if arr is not None:
        for da in (0.05, -0.05, 0.1, -0.1):
            if time.time() - start_time > LIMIT:
                break
            aspect = (w / h) + da
            if aspect <= 0:
                continue
            arr2, s2, _, _ = _solve_aspect(aspect, start_time)
            _record(arr2, s2)

    # Phase 3: NM polish + full expand on the global best
    if best_arr is not None:
        p, pw, ph = _perimeter(best_arr)
        if p <= 4.0 + 1e-7:
            w, h = pw, ph
        else:
            w, h = 1.0, 1.0
        t_left = LIMIT - (time.time() - start_time)
        if t_left > 3.0:
            polished = _nm_polish(best_arr, w, h, t_left, maxiter=3000)
            if _is_valid(polished, w, h):
                best_arr = polished
            expanded = _full_expand(best_arr.copy(), w, h)
            if _is_valid(expanded, w, h):
                best_arr = expanded
            best_sum = float(np.sum(best_arr[:, 2]))

    # Final validity + perimeter guard
    if best_arr is not None:
        p, _, _ = _perimeter(best_arr)
        if p > 4.0 + 1e-7 or not _is_valid(best_arr):
            # scale to fit perimeter 4
            scale = 4.0 / p
            best_arr = best_arr.copy()
            best_arr[:, 0] *= scale
            best_arr[:, 1] *= scale
            best_arr[:, 2] *= scale
            if not _is_valid(best_arr):
                best_arr[:, 2] *= 0.9
    if best_arr is None:
        # fallback grid
        best_arr = np.zeros((N, 3))
        c = 0
        for i in range(5):
            for j in range(5):
                if c >= N:
                    break
                best_arr[c] = [(i + 0.5) / 5, (j + 0.5) / 5, 0.5 / 5 - 0.01]
                c += 1
    return best_arr
