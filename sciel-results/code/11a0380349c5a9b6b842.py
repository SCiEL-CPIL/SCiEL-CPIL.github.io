"""Circle packing in perimeter-4 rectangle: hex-grid initialization + softplus penalty optimization.
Combines the variational softplus objective with analytic gradients (from inspiration) and
structured hex-grid initialization with multiple aspect ratios, plus a Delaunay-based
neighbor-reduced validation for fast polish.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
EPS = 1e-9
TIME_LIMIT = 145.0  # safety margin before 150s kill


def _softplus(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)


def _softplus_grad(x, beta=100.0):
    return 1.0 / (1.0 + np.exp(-beta * x))


def _objective_and_grad(x, w, h, beta=100.0):
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)

    # Boundary penalties
    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # Pairwise overlap penalties
    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        obj += np.sum(_softplus(v, beta))
        g = _softplus_grad(v, beta)

        grad[3 * i] += -np.sum(g * dx / d)
        grad[3 * i + 1] += -np.sum(g * dy / d)
        grad[3 * i + 2] += np.sum(g)

        idx = np.arange(i + 1, N)
        grad[3 * idx] += g * dx / d
        grad[3 * idx + 1] += g * dy / d
        grad[3 * idx + 2] += g

    grad[2::3] += -1.0
    return obj, grad


def _strictly_valid(circles, w, h, tol=1e-8):
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d2 = dx * dx + dy * dy
        need = rs[i] + rs[i + 1:] - tol
        if np.any(d2 < need * need):
            return False
    return True


def _delaunay_neighbors(centers):
    try:
        tri = Delaunay(centers)
        nbrs = [set() for _ in range(N)]
        for i, j, k in tri.simplices:
            nbrs[i].add(j)
            nbrs[i].add(k)
            nbrs[j].add(i)
            nbrs[j].add(k)
            nbrs[k].add(i)
            nbrs[k].add(j)
        return [list(s) for s in nbrs]
    except Exception:
        return [list(range(N)) for _ in range(N)]


def _fast_valid(circles, w, h, neighbors, tol=1e-9):
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        for j in neighbors[i]:
            if j <= i:
                continue
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[j] - tol
            if d2 < need * need:
                return False
    return True


def _hex_init(w, h, seed):
    """Generate hex-packed initial configurations with various row layouts."""
    rng = random.Random(seed)
    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3],
        [4, 4, 3, 4, 4, 2],
        [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3],
        [5, 4, 4, 4, 4],
        [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5],
        [4, 4, 5, 4, 4],
        [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1],
        [1, 5, 5, 5, 5],
        [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3],
        [4, 4, 4, 4, 4, 1],
        [1, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 5],
        [5, 4, 4, 4, 4],
        [3, 4, 4, 5, 5],
        [5, 5, 4, 4, 3],
    ]
    best = None
    best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        # Estimate radius from horizontal and vertical constraints
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        circles = np.array(circles)
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()
    if best is None:
        # Fallback: grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best = np.array(circles)
    return best


def _grid_init(w, h, seed):
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    circles = []
    for i in range(N):
        col = i % a
        row = i // a
        r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        x = (col + 0.5) * w / a
        y = (row + 0.5) * h / b
        circles.append([x, y, r])
    return np.array(circles)


def _random_init(w, h, seed):
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.2
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    best_val = -1e10
    best_result = None
    best_w = 1.0
    best_h = 1.0

    # Try multiple aspect ratios (w/h)
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5, 0.9, 1.1, 0.75, 1.25]

    for ar in aspect_ratios:
        if time.time() > deadline:
            break

        # perimeter = 2(w+h) = 4 => w+h = 2. If w/h = ar, then h = w/ar, w(1+1/ar)=2
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)

        # Generate multiple initializations
        inits = []
        for seed in range(5):
            circ = _hex_init(w, h, seed)
            inits.append(circ)
        for seed in range(2):
            circ = _grid_init(w, h, seed)
            inits.append(circ)
        for seed in range(1):
            circ = _random_init(w, h, seed)
            inits.append(circ)

        for circ in inits:
            if time.time() > deadline:
                break

            x0 = circ.flatten().copy()

            # Phase 1: Coarse optimization with moderate beta
            try:
                res = minimize(
                    lambda v: _objective_and_grad(v, w, h, beta=50.0)[0],
                    x0,
                    jac=lambda v: _objective_and_grad(v, w, h, beta=50.0)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'gtol': 1e-5, 'maxfun': 8000}
                )
            except Exception:
                continue

            if time.time() > deadline:
                break

            # Phase 2: Fine optimization with high beta
            try:
                res2 = minimize(
                    lambda v: _objective_and_grad(v, w, h, beta=200.0)[0],
                    res.x,
                    jac=lambda v: _objective_and_grad(v, w, h, beta=200.0)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 250, 'gtol': 1e-7, 'maxfun': 12000}
                )
            except Exception:
                res2 = res

            if time.time() > deadline:
                break

            # Phase 3: Ultra-fine polish with very high beta
            try:
                res3 = minimize(
                    lambda v: _objective_and_grad(v, w, h, beta=500.0)[0],
                    res2.x,
                    jac=lambda v: _objective_and_grad(v, w, h, beta=500.0)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'gtol': 1e-8, 'maxfun': 8000}
                )
            except Exception:
                res3 = res2

            # Validate
            circles = res3.x.reshape(N, 3)
            if _strictly_valid(circles, w, h, tol=1e-7):
                val = float(np.sum(circles[:, 2]))
                if val > best_val:
                    best_val = val
                    best_result = circles.copy()
                    best_w = w
                    best_h = h

    # If we have a valid result, do a final ultra-polish using Delaunay-reduced checks
    if best_result is not None and time.time() < deadline - 5.0:
        # Final high-precision polish on the best result
        x_best = best_result.flatten().copy()
        try:
            res_final = minimize(
                lambda v: _objective_and_grad(v, best_w, best_h, beta=1000.0)[0],
                x_best,
                jac=lambda v: _objective_and_grad(v, best_w, best_h, beta=1000.0)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-10, 'maxfun': 10000}
            )
            circles_final = res_final.x.reshape(N, 3)
            if _strictly_valid(circles_final, best_w, best_h, tol=1e-7):
                val_final = float(np.sum(circles_final[:, 2]))
                if val_final > best_val:
                    best_val = val_final
                    best_result = circles_final.copy()
        except Exception:
            pass

    # Fallback: return a valid grid packing
    if best_result is None:
        # Square packing
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        circles = []
        for i in range(N):
            col = i % side
            row = i // side
            r = 0.5 / side - 0.001
            x = (col + 0.5) * w / side
            y = (row + 0.5) * h / side
            circles.append([x, y, r])
        best_result = np.array(circles)
        best_val = float(np.sum(circles[:, 2]))
        best_w = w
        best_h = h

    # Ensure output format: (n, 3) array of (x, y, r)
    return best_result
