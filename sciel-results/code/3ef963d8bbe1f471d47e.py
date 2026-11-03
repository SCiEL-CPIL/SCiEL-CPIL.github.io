"""Optimized packing of 300 equal circles in a unit circle."""
import numpy as np
import time

N = 300
TIME_LIMIT = 118.0  # evaluator kills at 150s; keep a safety margin


def _hex_init(n=N):
    """Dense hexagonal lattice clipped to the unit disk, keep n closest to center."""
    # Best known r ~ 0.05315 -> spacing ~ 0.1063. Use a slightly finer grid.
    spacing = 0.095
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        # fallback: enlarge grid
        return _hex_init_fallback(n)
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _hex_init_fallback(n=N):
    spacing = 0.08
    k = int(1.2 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _score(pts):
    """Achieved common radius (min over pairs and boundary)."""
    n = len(pts)
    # pairwise distances
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _optimize(pts, start_time, time_limit):
    """Soft-min gradient ascent optimizer. Returns best pts."""
    n = len(pts)
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Beta controls the "sharpness" of the soft-min.
    # Start low to allow global movement, increase to focus on local minima.
    beta = 50.0
    step_size = 0.01
    min_step = 1e-6
    max_iter = 10000
    
    it = 0
    while it < max_iter:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # 1. Compute distances
        # Pairwise
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Boundary
        rad = np.hypot(pts[:, 0], pts[:, 1])
        b = 1.0 - rad
        
        # 2. Compute soft-min weights
        # We want to maximize min(d, b).
        # Soft min of a set of values v_k is -1/beta * log(sum(exp(-beta * v_k)))
        # We want to maximize this, so we minimize -soft_min.
        # Gradient of soft_min wrt x is sum( w_k * grad(v_k) ) where w_k = exp(-beta*v_k) / sum(exp(-beta*v_j))
        
        with np.errstate(over='ignore', under='ignore'):
            # Pairwise weights
            # d has inf on diagonal. exp(-inf) = 0.
            w_pair = np.exp(-beta * d)
            np.fill_diagonal(w_pair, 0.0)
            
            # Boundary weights
            w_bound = np.exp(-beta * b)
            
            # Total weight sum
            sum_w = np.sum(w_pair) + np.sum(w_bound)
            
            if sum_w < 1e-10:
                break
                
            # Gradients
            # Grad of d_ij wrt x_i is (x_i - x_j) / d_ij
            # Grad of b_i wrt x_i is -x_i / r_i
            
            with np.errstate(divide='ignore', invalid='ignore'):
                unit = diff / d[:, :, None]
            unit = np.where(np.isfinite(unit), unit, 0.0)
            
            # Weighted sum of unit vectors
            # w_pair is (n,n). unit is (n,n,2).
            # F_pair = sum_j w_pair[i,j] * unit[i,j]
            F_pair = np.sum(w_pair[:, :, None] * unit, axis=1)
            
            # Force from boundary:
            # F_bound_i = w_bound[i] * (-x_i / r_i)
            with np.errstate(divide='ignore', invalid='ignore'):
                inward = -pts / rad[:, None]
            inward = np.where(np.isfinite(inward), inward, 0.0)
            F_bound = w_bound[:, None] * inward
            
            # Total Gradient (direction of increasing soft-min)
            grad = (F_pair + F_bound) / sum_w
            
            # Update
            # We are maximizing, so add gradient.
            # We need to ensure we don't step too far.
            # Normalize gradient to have unit magnitude?
            # Or just use a fixed step size?
            # Let's normalize to avoid huge steps if weights are large.
            g_norm = np.linalg.norm(grad, axis=1)
            max_g = np.max(g_norm)
            if max_g > 0:
                grad = grad * (step_size / max_g)
            
            new_pts = pts + grad
            
            # Enforce boundary constraint (hard)
            new_rad = np.hypot(new_pts[:, 0], new_pts[:, 1])
            out = new_rad > 1.0
            if np.any(out):
                new_pts[out] = new_pts[out] / new_rad[out, None]
            
            # Check improvement
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts
                # Increase beta slightly to focus more
                beta *= 1.05
                # Reduce step size slightly
                step_size *= 0.99
            else:
                # No improvement, reduce step size and maybe increase beta
                step_size *= 0.95
                beta *= 1.1
                if step_size < min_step:
                    break
            
            it += 1
            if it % 50 == 0:
                # Re-evaluate best score to be sure
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()

    return best_pts, best_score


def construct():
    start_time = time.time()
    pts = _hex_init(N)
    # ensure valid start: clip inside
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]

    base_score = _score(pts)
    best_pts = pts.copy()
    best_score = base_score

    try:
        opt_pts, opt_score = _optimize(pts, start_time, TIME_LIMIT)
        if opt_score > best_score:
            best_pts = opt_pts
            best_score = opt_score
    except Exception:
        pass

    # final safety: ensure finite, inside, shape (N,2)
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        # rebuild
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]

    return best_pts
