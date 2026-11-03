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


def _soft_min_value(pts, beta=50.0):
    """
    Smooth surrogate for the min radius using log-sum-exp.
    Returns (value, gradient) where value approximates min(r_pair, r_bound)
    and gradient is d(value)/d(pts).
    
    We minimize -value, so gradient of objective = -gradient of value.
    """
    n = len(pts)
    
    # Pairwise distances
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, 0.0)
    
    # r_pair_ij = d_ij / 2
    # We want soft_min over all pairs of d_ij/2
    # log-sum-exp: log(sum(exp(beta * (d_ij/2)))) / beta approximates max, so we use -1/beta * log(sum(exp(-beta * d_ij/2)))
    
    # For numerical stability, work with d2
    # d_ij = sqrt(d2_ij)
    # We want to approximate min_i<j d_ij/2
    # soft_min = -1/beta * log( sum_{i<j} exp(-beta * d_ij/2) )
    
    # But computing exp for all pairs is expensive. Let's use a simpler approach:
    # Compute the actual min and use it as a baseline, then add a smooth penalty.
    
    # Actually, let's use the epigraph formulation with a smooth penalty:
    # Objective: minimize -t
    # Constraints: d_ij >= 2t, |c_i| <= 1-t
    # Smooth penalty: sum of soft violations
    
    # Simpler: use the gradient of the min directly with a smoothing parameter
    # For pairs: if d_ij is the min, gradient w.r.t. c_i is -unit_ij, w.r.t. c_j is +unit_ij
    # Smooth version: weight each pair by softmax over (target - d_ij)
    
    # Let's use a different smooth surrogate:
    # f(pts) = -1/beta * log( sum_{i<j} exp(-beta * (d_ij - target) / 2) * exp(-beta * (1 - |c_i| - target)) )
    # This is getting complex. Let me use a simpler approach:
    
    # Compute pairwise distances
    d = np.sqrt(np.maximum(d2, 0.0))
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances
    rad = np.hypot(pts[:, 0], pts[:, 1])
    r_bound = 1.0 - rad  # distance to boundary
    
    # Soft min of pairwise distances / 2
    # Use log-sum-exp with negative sign for min
    # soft_min_pair = -1/beta * log(sum(exp(-beta * d/2)))
    # But this is over all pairs, which is O(n^2)
    
    # For efficiency, let's compute the gradient of a smooth approximation:
    # We'll use the fact that near the min, the gradient is dominated by the closest pairs.
    
    # Simple approach: compute the actual min and its gradient, but smooth the selection
    min_pair_val = np.min(d) / 2.0
    min_bound_val = np.min(r_bound)
    
    # Determine which is the bottleneck
    if min_pair_val < min_bound_val:
        # Pairwise is the bottleneck
        # Find all pairs with distance close to min
        threshold = min_pair_val * 2.0 + 1e-6
        close_mask = d < threshold
        np.fill_diagonal(close_mask, False)
        
        # Count how many close pairs
        n_close = np.sum(close_mask)
        if n_close == 0:
            return 0.0, np.zeros_like(pts)
        
        # Gradient: for each close pair (i,j), gradient w.r.t. c_i is -(c_i - c_j)/d_ij * w_ij
        # where w_ij is a weight
        # Use uniform weight for simplicity (or softmax)
        weights = np.where(close_mask, 1.0 / n_close, 0.0)
        
        diff = pts[:, None, :] - pts[None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            unit = diff / d[:, :, None]
        unit = np.where(np.isfinite(unit), unit, 0.0)
        
        # Gradient of min_pair w.r.t. pts[i] = sum_j weights[i,j] * (-unit[i,j])
        grad = -np.sum(weights[:, :, None] * unit, axis=1)
        
        return min_pair_val, grad
    else:
        # Boundary is the bottleneck
        # Find points closest to boundary
        threshold = min_bound_val + 1e-6
        close_mask = r_bound < threshold
        n_close = np.sum(close_mask)
        if n_close == 0:
            return 0.0, np.zeros_like(pts)
        
        weights = np.where(close_mask, 1.0 / n_close, 0.0)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            inward = -pts / rad[:, None]
        inward = np.where(np.isfinite(inward), inward, 0.0)
        
        # Gradient of min_bound w.r.t. pts[i] = weights[i] * (-inward[i])
        # because r_bound[i] = 1 - |c_i|, dr_bound/dc_i = -c_i/|c_i|
        grad = -weights[:, None] * inward
        
        return min_bound_val, grad


def _optimize(pts, start_time, time_limit):
    """Gradient-based optimizer using smooth surrogate. Returns best pts."""
    n = len(pts)
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _score(pts)

    # Adaptive step size
    step = 0.002
    it = 0
    no_improve = 0
    max_no_improve = 500
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if no_improve > max_no_improve:
            break

        # --- compute smooth gradient ---
        # Compute pairwise distances
        d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
        np.fill_diagonal(d2, 0.0)
        d = np.sqrt(np.maximum(d2, 0.0))
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        rad = np.hypot(pts[:, 0], pts[:, 1])
        r_bound = 1.0 - rad
        
        min_pair_val = np.min(d) / 2.0
        min_bound_val = np.min(r_bound)
        current_score = min(min_pair_val, min_bound_val)
        
        # Compute gradient of smooth surrogate
        # Use softmax weighting over close constraints
        beta = 30.0  # smoothing parameter
        
        # Pairwise gradient contribution
        # Weight each pair by softmax over (target - d_ij) where target = 2*current_score
        target = 2.0 * current_score
        pair_violation = np.maximum(0.0, target - d)
        
        # Softmax weights (numerically stable)
        max_v = np.max(pair_violation)
        if max_v > 0:
            exp_v = np.exp(beta * (pair_violation - max_v))
            exp_v[~np.isfinite(exp_v)] = 0.0
            sum_exp = np.sum(exp_v)
            if sum_exp > 0:
                pair_weights = exp_v / sum_exp
            else:
                pair_weights = np.zeros_like(d)
        else:
            pair_weights = np.zeros_like(d)
        np.fill_diagonal(pair_weights, 0.0)
        
        diff = pts[:, None, :] - pts[None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            unit = diff / d[:, :, None]
        unit = np.where(np.isfinite(unit), unit, 0.0)
        
        # Gradient of min_pair w.r.t. pts[i] = -sum_j pair_weights[i,j] * unit[i,j]
        grad_pair = -np.sum(pair_weights[:, :, None] * unit, axis=1)
        
        # Boundary gradient contribution
        bound_violation = np.maximum(0.0, current_score - r_bound)
        max_bv = np.max(bound_violation)
        if max_bv > 0:
            exp_bv = np.exp(beta * (bound_violation - max_bv))
            exp_bv[~np.isfinite(exp_bv)] = 0.0
            sum_exp_bv = np.sum(exp_bv)
            if sum_exp_bv > 0:
                bound_weights = exp_bv / sum_exp_bv
            else:
                bound_weights = np.zeros_like(r_bound)
        else:
            bound_weights = np.zeros_like(r_bound)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            inward = -pts / rad[:, None]
        inward = np.where(np.isfinite(inward), inward, 0.0)
        
        # Gradient of min_bound w.r.t. pts[i] = -bound_weights[i] * inward[i]
        grad_bound = -bound_weights[:, None] * inward
        
        # Combine gradients
        grad = grad_pair + grad_bound
        
        # Add mild centering to prevent drift
        grad += -pts * 0.01
        
        # --- update ---
        fmag = np.linalg.norm(grad, axis=1)
        max_fm = np.max(fmag) if len(fmag) else 1.0
        if max_fm > 0:
            grad = grad * (step / max_fm)
        
        pts = pts + grad
        
        # Clamp inside boundary (safety)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        out = rad > 1.0
        if np.any(out):
            pts[out] = pts[out] / rad[out, None]
        
        it += 1
        
        # Recompute score periodically
        if it % 10 == 0:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 10
        
        # Adapt step size
        if it % 100 == 0:
            if no_improve > 200:
                step *= 0.9
            elif no_improve < 50:
                step *= 1.05
            step = min(step, 0.01)
            step = max(step, 0.0001)

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
