"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Delaunay

N = 21
TIME_LIMIT = 145.0

def _softplus(x, beta=100.0):
    """Numerically stable softplus: log(1+exp(beta*x))/beta"""
    xb = beta * x
    return np.where(xb > 0, 
                    x + np.log1p(np.exp(-xb)) / beta, 
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=100.0):
    """Gradient of softplus: sigmoid(beta*x)"""
    xb = beta * x
    return np.where(xb > 0, 
                    1.0 / (1.0 + np.exp(-xb)), 
                    np.exp(xb) / (1.0 + np.exp(xb)))

def _objective_and_grad(x, w, h, beta=100.0, var_weight=0.0, contact_edges=None):
    """
    Objective: -sum(r) + penalties for violations.
    x: [x1, y1, r1, x2, y2, r2, ...]
    contact_edges: list of (i,j) pairs to enforce non-overlap (if None, use all pairs)
    """
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)
    
    # 1. Boundary Constraints
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

    # 2. Pairwise Non-overlap Constraints
    if contact_edges is not None:
        edge_list = contact_edges
    else:
        iu = np.triu_indices(N, k=1)
        edge_list = list(zip(iu[0], iu[1]))
    
    for i, j in edge_list:
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        r_sum = rs[i] + rs[j]
        v = r_sum - d
        
        pen = _softplus(v, beta)
        obj += pen
        
        g = _softplus_grad(v, beta)
        
        grad[3*i]   += -g * dx / d
        grad[3*i+1] += -g * dy / d
        grad[3*i+2] +=  g
        
        grad[3*j]   +=  g * dx / d
        grad[3*j+1] +=  g * dy / d
        grad[3*j+2] +=  g

    # 3. Variance Penalty (Equalization Pressure)
    if var_weight > 0.0:
        r_mean = np.mean(rs)
        diff = rs - r_mean
        obj += var_weight * np.sum(diff * diff)
        grad[2::3] += 2.0 * var_weight * diff

    # 4. Objective gradient for -sum(r)
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
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    d2 = dx * dx + dy * dy
    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    iu = np.triu_indices(n, k=1)
    d2_upper = d2[iu]
    r_sum_upper = r_sum[iu]
    if np.any(d2_upper < (r_sum_upper - tol) ** 2):
        return False
    return True

def _get_delaunay_edges(circles):
    """Get Delaunay triangulation edges for contact graph."""
    points = circles[:, :2]
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                j = (i + 1) % 3
                edge = tuple(sorted((simplex[i], simplex[j])))
                edges.add(edge)
        return list(edges)
    except Exception:
        iu = np.triu_indices(N, k=1)
        return list(zip(iu[0], iu[1]))

def _force_directed_step(circles, w, h, iterations=50, dt=0.001):
    """Physics-based force-directed refinement to escape local minima."""
    circles = circles.copy()
    for _ in range(iterations):
        forces = np.zeros_like(circles)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Repulsion between overlapping or close circles
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy + 1e-12)
                r_sum = rs[i] + rs[j]
                if d < r_sum * 1.5:
                    # Strong repulsion when overlapping, weaker when close
                    overlap = r_sum - d
                    if overlap > 0:
                        f = 10.0 * overlap
                    else:
                        f = 0.1 * (r_sum - d)
                    fx = f * dx / d
                    fy = f * dy / d
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Attraction to keep circles in bounds
        for i in range(N):
            if xs[i] - rs[i] < 0.05:
                forces[i, 0] += 5.0 * (0.05 - (xs[i] - rs[i]))
            if xs[i] + rs[i] > w - 0.05:
                forces[i, 0] -= 5.0 * ((xs[i] + rs[i]) - (w - 0.05))
            if ys[i] - rs[i] < 0.05:
                forces[i, 1] += 5.0 * (0.05 - (ys[i] - rs[i]))
            if ys[i] + rs[i] > h - 0.05:
                forces[i, 1] -= 5.0 * ((ys[i] + rs[i]) - (h - 0.05))
        
        # Apply forces with damping
        circles[:, 0] += dt * forces[:, 0]
        circles[:, 1] += dt * forces[:, 1]
        
        # Clip to bounds
        circles[:, 0] = np.clip(circles[:, 0], rs, w - rs)
        circles[:, 1] = np.clip(circles[:, 1], rs, h - rs)
    
    return circles

def _hex_init(w, h, seed):
    """Generate hex-packed initial configurations."""
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
        [6, 5, 5, 5],
        [5, 6, 5, 5],
        [5, 5, 6, 5],
        [5, 5, 5, 6],
    ]
    best = None
    best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        
        est_r_h = w / (max_cols * 2.0 + 0.5)
        est_r_v = h / ((n_rows - 1) * np.sqrt(3) + 2.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
            
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = r_est + row_idx * r_est * np.sqrt(3)
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

def _multi_phase_optimize(circles, w, h, deadline):
    """Multi-phase optimization: force-directed + gradient with contact graph refinement."""
    best_circles = circles.copy()
    best_val = np.sum(circles[:, 2]) if _strictly_valid(circles, w, h) else -np.inf
    
    # Phase 1: Force-directed relaxation
    if time.time() < deadline - 10:
        fd_circles = _force_directed_step(circles, w, h, iterations=30, dt=0.002)
        if _strictly_valid(fd_circles, w, h):
            val = np.sum(fd_circles[:, 2])
            if val > best_val:
                best_val = val
                best_circles = fd_circles.copy()
    
    # Phase 2: Gradient optimization with full constraints
    if time.time() < deadline - 8:
        try:
            res = minimize(
                lambda p: _objective_and_grad(p, w, h, 100.0, 0.0),
                best_circles.flatten(),
                jac=True,
                method='SLSQP',
                bounds=[(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)] * N,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            p_opt = res.x.reshape(-1, 3)
            if _strictly_valid(p_opt, w, h):
                val = np.sum(p_opt[:, 2])
                if val > best_val:
                    best_val = val
                    best_circles = p_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Contact-graph-based optimization (faster, more focused)
    if time.time() < deadline - 6:
        edges = _get_delaunay_edges(best_circles)
        try:
            res2 = minimize(
                lambda p: _objective_and_grad(p, w, h, 150.0, 0.01, edges),
                best_circles.flatten(),
                jac=True,
                method='SLSQP',
                bounds=[(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)] * N,
                options={'maxiter': 150, 'ftol': 1e-12}
            )
            p_opt2 = res2.x.reshape(-1, 3)
            if _strictly_valid(p_opt2, w, h):
                val2 = np.sum(p_opt2[:, 2])
                if val2 > best_val:
                    best_val = val2
                    best_circles = p_opt2.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish with high beta
    if time.time() < deadline - 4:
        try:
            res3 = minimize(
                lambda p: _objective_and_grad(p, w, h, 500.0, 0.0),
                best_circles.flatten(),
                jac=True,
                method='SLSQP',
                bounds=[(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)] * N,
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            p_opt3 = res3.x.reshape(-1, 3)
            if _strictly_valid(p_opt3, w, h):
                val3 = np.sum(p_opt3[:, 2])
                if val3 > best_val:
                    best_val = val3
                    best_circles = p_opt3.copy()
        except Exception:
            pass
    
    return best_circles, best_val

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
    xs = rng.uniform(0.1*w, 0.9*w, N)
    ys = rng.uniform(0.1*h, 0.9*h, N)
    rs = np.full(N, 0.08 * min(w, h))
    return np.column_stack([xs, ys, rs])

def _scale_to_perim(circles, target_perim=4.0):
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    perim = 2.0 * ((max_x - min_x) + (max_y - min_y))
    if perim <= target_perim + 1e-9:
        return circles, perim
    scale = target_perim / perim
    cx = np.mean(xs)
    cy = np.mean(ys)
    xs_new = cx + (xs - cx) * scale
    ys_new = cy + (ys - cy) * scale
    rs_new = rs * scale
    return np.column_stack([xs_new, ys_new, rs_new]), target_perim

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    random.seed(42)
    np.random.seed(42)
    
    best_value = -np.inf
    best_result = None
    
    aspect_ratios = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 20:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        
        # Generate initializations
        inits = []
        
        # Hex
        hex_circ = _hex_init(w, h, 42)
        inits.append(hex_circ)
        
        # Grid
        grid_circ = _grid_init(w, h, 42)
        inits.append(grid_circ)
        
        # Random
        for seed in [1, 2, 3, 42, 100]:
            rand_circ = _random_init(w, h, seed)
            inits.append(rand_circ)
        
        for init_circ in inits:
            if time.time() > deadline - 15:
                break
            
            # Multi-phase optimization
            optimized_circ, opt_val = _multi_phase_optimize(init_circ, w, h, deadline)
            
            if opt_val > best_value:
                best_value = opt_val
                best_result = optimized_circ.copy()
        
        # Polish best result found so far for this aspect ratio
        if best_result is not None and time.time() < deadline - 8:
            try:
                res_polish = minimize(
                    lambda p: _objective_and_grad(p, w, h, 500.0, 0.0),
                    best_result.flatten(),
                    jac=True,
                    method='SLSQP',
                    bounds=[(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)] * N,
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                p_polish = res_polish.x.reshape(-1, 3)
                if _strictly_valid(p_polish, w, h):
                    val_polish = np.sum(p_polish[:, 2])
                    if val_polish > best_value:
                        best_value = val_polish
                        best_result = p_polish.copy()
            except Exception:
                pass
    
    # Final refinement: local search with perturbation
    if best_result is not None and time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            perturbed = best_result + np.random.randn(*best_result.shape) * 0.002
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 2.0)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 2.0)
            perturbed[:, 2] = np.clip(perturbed[:, 2], 1e-4, 0.5)
            
            # Quick validity check
            if _strictly_valid(perturbed, 1.0, 1.0):
                # Re-optimize from perturbed
                try:
                    res_pert = minimize(
                        lambda p: _objective_and_grad(p, 1.0, 1.0, 100.0, 0.0),
                        perturbed.flatten(),
                        jac=True,
                        method='SLSQP',
                        bounds=[(0, 1.0)] * N + [(0, 1.0)] * N + [(1e-4, 0.5)] * N,
                        options={'maxiter': 100, 'ftol': 1e-12}
                    )
                    p_pert = res_pert.x.reshape(-1, 3)
                    if _strictly_valid(p_pert, 1.0, 1.0):
                        val_pert = np.sum(p_pert[:, 2])
                        if val_pert > best_value:
                            best_value = val_pert
                            best_result = p_pert.copy()
                except Exception:
                    pass
    
    # Return result
    if best_result is None:
        # Fallback: simple grid
        xs, ys = _grid_init(1.0, 1.0, 42)[:, :2].T
        r = 0.08
        best_result = np.column_stack([xs, ys, np.full(N, r)])
    
    # Scale to ensure perimeter <= 4
    best_result, _ = _scale_to_perim(best_result, 4.0)
    
    return best_result
