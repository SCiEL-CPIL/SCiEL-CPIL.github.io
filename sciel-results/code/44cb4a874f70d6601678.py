"""
Circle packing in perimeter-4 rectangle:
Optimized gradient-based search with adaptive penalties and aggressive polishing.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
EPS = 1e-9
START_TIME = time.time()

def _get_remaining_time(deadline):
    return deadline - time.time()

def _softplus(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=100.0):
    return 1.0 / (1.0 + np.exp(-beta * x))

def _objective_and_grad(x, w, h, beta=100.0, pairs=None, var_weight=0.0):
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)
    
    # Equal-radius variance penalty to encourage uniformity
    if var_weight > 0:
        mean_r = np.mean(rs)
        var_r = np.var(rs)
        obj += var_weight * var_r
        grad[2::3] += 2.0 * var_weight * (rs - mean_r) / N

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
    if pairs is not None:
        for (i, j) in pairs:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[j] - d
            obj += _softplus(v, beta)
            g = _softplus_grad(v, beta)
            grad[3 * i] += -g * dx / d
            grad[3 * i + 1] += -g * dy / d
            grad[3 * i + 2] += g
            grad[3 * j] += g * dx / d
            grad[3 * j + 1] += g * dy / d
            grad[3 * j + 2] += g
    else:
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

def _strictly_valid(circles, w, h, tol=1e-7):
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

def _delaunay_pairs(circles, w, h, margin=0.1):
    centers = circles[:, :2]
    rs = circles[:, 2]
    try:
        tri = Delaunay(centers)
        pairs_set = set()
        for simplex in tri.simplices:
            for idx in range(3):
                for jdx in range(idx + 1, 3):
                    i, j = simplex[idx], simplex[jdx]
                    if i > j:
                        i, j = j, i
                    pairs_set.add((i, j))
        # Add close pairs not in Delaunay
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) in pairs_set:
                    continue
                d = np.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
                if d < rs[i] + rs[j] + margin:
                    pairs_set.add((i, j))
        return list(pairs_set)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _fast_valid(circles, w, h, neighbors, tol=1e-8):
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

def _delaunay_neighbors(centers):
    try:
        tri = Delaunay(centers)
        nbrs = [set() for _ in range(N)]
        for i, j, k in tri.simplices:
            nbrs[i].add(j); nbrs[i].add(k)
            nbrs[j].add(i); nbrs[j].add(k)
            nbrs[k].add(i); nbrs[k].add(j)
        return [list(s) for s in nbrs]
    except Exception:
        return [list(range(N)) for _ in range(N)]

def _hex_init(w, h, seed):
    """
    Generate hexagonal lattice initialization.
    """
    rng = random.Random(seed)
    # Try different row counts to find a valid hex packing
    # For 21 circles, common patterns are 4-5-4-5-3 or 5-4-5-4-3 etc.
    row_patterns = [
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
    ]
    
    best_circles = None
    best_val = -np.inf

    for row_counts in row_patterns:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        
        # Estimate radius based on width and height
        # Horizontal: max_cols * 2r <= w -> r <= w / (2*max_cols)
        # Vertical: (n_rows - 1) * sqrt(3) * r + 2r <= h -> r <= h / (2 + (n_rows-1)*sqrt(3))
        est_r_w = w / (2.0 * max_cols * 1.05) # 5% safety
        est_r_h = h / (2.0 + (n_rows - 1) * np.sqrt(3) * 1.05)
        r_est = min(est_r_w, est_r_h)
        
        if r_est < 0.05:
            continue

        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = r_est + row_idx * np.sqrt(3) * r_est
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2.0
            for col_idx in range(n_cols):
                x = center_x - row_width / 2.0 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        
        circles = np.array(circles)
        if _strictly_valid(circles, w, h, tol=1e-5):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best_circles = circles.copy()

    if best_circles is not None:
        return best_circles

    # Fallback: Grid initialization
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
    r0 = min(w, h) / 2.0 * 0.25
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles

def _optimize_phase(circles0, w, h, deadline, beta_start, beta_end, max_iter, var_weight=0.0):
    """
    Multi-phase optimization with ramping beta.
    """
    x0 = circles0.flatten()
    r_max = min(w, h) / 2.0
    bounds = []
    for _ in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((1e-6, r_max))

    pairs = _delaunay_pairs(circles0, w, h, margin=0.1)
    
    def obj_fn(p, beta):
        return _objective_and_grad(p, w, h, beta, pairs=pairs, var_weight=var_weight)

    try:
        res = minimize(
            lambda p: obj_fn(p, beta_start),
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-7},
        )
        if time.time() > deadline:
            return None, -np.inf
            
        circles = res.x.reshape(-1, 3)
        
        # Second phase with higher beta if time remains
        if time.time() < deadline - 5:
            res2 = minimize(
                lambda p: obj_fn(p, beta_end),
                circles.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-9},
            )
            if time.time() < deadline:
                circles = res2.x.reshape(-1, 3)

        if _strictly_valid(circles, w, h):
            return circles, float(np.sum(circles[:, 2]))
        return None, -np.inf
    except Exception:
        return None, -np.inf

def _polish(circles, w, h, deadline):
    """
    Iterative radius expansion with Delaunay-based validity checks.
    Uses adaptive step size.
    """
    circles = circles.copy()
    centers = circles[:, :2]
    neighbors = _delaunay_neighbors(centers)
    step_size = 1.01
    min_step = 1.001
    
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            old_r = circles[i, 2]
            # Try to increase radius
            for _attempt in range(3):
                new_r = old_r * step_size
                if new_r > min(w, h) / 2.0:
                    break
                circles[i, 2] = new_r
                if _fast_valid(circles, w, h, neighbors, tol=1e-8):
                    improved = True
                    break
                else:
                    circles[i, 2] = old_r
            
            if not improved:
                # If no improvement in this pass, reduce step size
                step_size = max(min_step, step_size * 0.9)
        
        if not improved:
            break
            
        if _ % 5 == 0:
            centers = circles[:, :2]
            neighbors = _delaunay_neighbors(centers)

    # Final strict validity check and binary search correction if needed
    if not _strictly_valid(circles, w, h, tol=1e-7):
        for i in range(N):
            lo, hi = 1e-6, circles[i, 2]
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                circles[i, 2] = mid
                if _strictly_valid(circles, w, h, tol=1e-7):
                    lo = mid
                else:
                    hi = mid
            circles[i, 2] = lo
    
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 35s buffer for safety

    best_circles = None
    best_value = -np.inf
    best_w = best_h = None

    # Generate aspect ratios to test
    # Focus on ratios where hexagonal packing is efficient
    aspect_ratios = []
    # Square-ish
    for ar in [0.8, 0.9, 1.0, 1.1, 1.2]:
        aspect_ratios.append(ar)
    # Elongated
    for ar in [1.4, 1.6, 1.8, 2.0, 2.2]:
        aspect_ratios.append(ar)
    
    # Sort by priority (1.0 and 1.5 are often good starting points for hex)
    # We'll just iterate and let the optimizer handle it, but ensure we try square early.
    
    seed_counter = 0

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        
        # Skip if dimensions are too small
        if w < 0.5 or h < 0.5:
            continue

        # Initializations
        inits = [
            _hex_init(w, h, seed_counter),
            _grid_init(w, h, seed_counter + 100),
            _random_init(w, h, seed_counter + 200)
        ]
        
        for init_idx, circles0 in enumerate(inits):
            if time.time() > deadline - 5:
                break
            
            # Adaptive optimization: Start with lower beta, finish with higher
            # Fewer iterations for random init to save time? No, keep consistent.
            beta_start = 50.0
            beta_end = 200.0
            max_iter = 150
            
            circles, val = _optimize_phase(circles0, w, h, deadline, beta_start, beta_end, max_iter)
            
            if circles is not None:
                circles = _polish(circles, w, h, deadline)
                if _strictly_valid(circles, w, h, tol=1e-7):
                    val = float(np.sum(circles[:, 2]))
                    if val > best_value:
                        best_value = val
                        best_circles = circles.copy()
                        best_w, best_h = w, h

        seed_counter += 50

    # Final high-precision polish on the best candidate
    if best_circles is not None and time.time() < deadline - 5:
        # One more optimization pass with very high beta
        x0 = best_circles.flatten()
        r_max = min(best_w, best_h) / 2.0
        bounds = []
        for _ in range(N):
            bounds.append((0.0, best_w))
            bounds.append((0.0, best_h))
            bounds.append((1e-6, r_max))
            
        pairs = _delaunay_pairs(best_circles, best_w, best_h, margin=0.05)
        
        try:
            res = minimize(
                lambda p: _objective_and_grad(p, best_w, best_h, 500.0, pairs=pairs, var_weight=0.5),
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10},
            )
            if time.time() < deadline:
                circles_final = res.x.reshape(-1, 3)
                if _strictly_valid(circles_final, best_w, best_h, tol=1e-7):
                    circles_final = _polish(circles_final, best_w, best_h, deadline)
                    if _strictly_valid(circles_final, best_w, best_h, tol=1e-7):
                        val_final = float(np.sum(circles_final[:, 2]))
                        if val_final > best_value:
                            best_value = val_final
                            best_circles = circles_final
        except Exception:
            pass

    # Fallback if nothing worked
    if best_circles is None:
        w, h = 1.0, 1.0
        circles = np.zeros((N, 3))
        a = 5
        b = 5
        # 21 circles in 5x5 grid
        # Fill grid
        count = 0
        for i in range(5):
            for j in range(5):
                if count < N:
                    r = min(w / 5.0, h / 5.0) * 0.95
                    circles[count, 0] = (i + 0.5) * w / 5.0
                    circles[count, 1] = (j + 0.5) * h / 5.0
                    circles[count, 2] = r
                    count += 1
        best_circles = circles
        best_value = float(np.sum(best_circles[:, 2]))

    return best_circles
