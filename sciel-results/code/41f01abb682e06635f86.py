"""Packing 50 equal circles in a unit square – optimized via soft-min + exact L-BFGS-B."""
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = 145.0  # seconds (safety margin before 150 s kill)

def _compute_radius(pts):
    """Exact radius: min over pairs of dist/2 and over walls of (half - max coord)."""
    # wall distances
    wall = np.minimum(HALF - np.abs(pts), axis=1)
    min_wall = wall.min()
    # pair distances
    d = cdist(pts, pts)
    # zero out diagonal
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_wall, min_pair)

def _soft_objective(pts, beta):
    """Smooth objective: -softmin of all constraints (pair dists and wall margins).
    Returns (value, grad). value is negative so minimization maximizes radius.
    """
    # wall margins: for each point, min over its 4 walls
    # wall_i = min(HALF - pts[i,0], HALF + pts[i,0], HALF - pts[i,1], HALF + pts[i,1])
    # We'll compute soft-min over all pair distances and all wall margins.
    
    # Pair distances (off-diagonal)
    d = cdist(pts, pts)
    # Zero out diagonal
    idx = np.arange(N)
    d[idx, idx] = np.inf
    
    # Wall margins for each point
    # For point i: min(HALF - x_i, HALF + x_i, HALF - y_i, HALF + y_i)
    # = min(HALF - |x_i|, HALF - |y_i|) ... but let's be explicit
    # Actually wall margin = min over the 4 sides
    # = HALF - max(|x_i|, |y_i|)
    wall_margins = HALF - np.max(np.abs(pts), axis=1)  # shape (N,)
    
    # Collect all "constraint values" we want to soft-minimize
    # Pair distances: d (N x N, off-diag inf)
    # Wall margins: wall_margins (N,)
    # We want softmin of all these values.
    
    # Flatten pair distances (only upper triangle to avoid double counting, but for softmin it doesn't matter much)
    # Use all off-diagonal
    pair_vals = d[d < np.inf]  # shape (N*(N-1),)
    all_vals = np.concatenate([pair_vals, wall_margins])  # shape (N*(N-1)+N,)
    
    # Soft-min: -logsumexp(-beta * vals) / beta
    # For numerical stability
    beta_val = beta
    scaled = -beta_val * all_vals
    # logsumexp
    max_s = scaled.max()
    logsumexp = max_s + np.log(np.exp(scaled - max_s).sum())
    softmin_val = -logsumexp / beta_val
    
    # Gradient: d(softmin)/d(pts)
    # softmin = -logsumexp(-beta * vals) / beta
    # d(softmin)/d(val_j) = (beta * exp(-beta*val_j)) / (sum_k exp(-beta*val_k))
    # = exp(scaled_j) / sum(exp(scaled_k))
    # where scaled_j = -beta * val_j
    exp_scaled = np.exp(scaled - max_s)
    weights = exp_scaled / exp_scaled.sum()  # shape (N*(N-1)+N,)
    
    # Now backpropagate weights to pts
    grad = np.zeros_like(pts)
    
    # Wall margins: wall_margins[i] = HALF - max(|x_i|, |y_i|)
    # d(wall_margins[i])/d(x_i) = -sign(x_i) if |x_i| > |y_i|, 0 otherwise (subgradient)
    # d(wall_margins[i])/d(y_i) = -sign(y_i) if |y_i| >= |x_i|, 0 otherwise
    abs_x = np.abs(pts[:, 0])
    abs_y = np.abs(pts[:, 1])
    # For wall margin, the active constraint is the one giving the min
    # wall_margins[i] = min(HALF - x_i, HALF + x_i, HALF - y_i, HALF + y_i)
    # = HALF - max(|x_i|, |y_i|)
    # d/dx_i of max(|x_i|, |y_i|) = sign(x_i) if |x_i| > |y_i|, 0.5*sign(x_i) if |x_i|==|y_i|, 0 if |x_i|<|y_i|
    # For simplicity, use subgradient: sign(x_i) if |x_i| >= |y_i| else 0
    # But this is only for the wall part. The weight for wall_margins[i] is weights[N*(N-1)+i]
    wall_weights = weights[N*(N-1):]  # shape (N,)
    
    # d(softmin)/d(pts[i,0]) from wall = -wall_weights[i] * d(wall_margins[i])/d(x_i)
    # d(wall_margins[i])/d(x_i) = -sign(x_i) if |x_i| >= |y_i| else 0
    x_active = abs_x >= abs_y
    y_active = ~x_active
    grad[:, 0] -= wall_weights * np.where(x_active, np.sign(pts[:, 0]), 0.0)
    grad[:, 1] -= wall_weights * np.where(y_active, np.sign(pts[:, 1]), 0.0)
    
    # Pair distances: d_{ij} = ||pts[i] - pts[j]||
    # d(d_{ij})/d(pts[i]) = (pts[i] - pts[j]) / d_{ij}
    # d(d_{ij})/d(pts[j]) = (pts[j] - pts[i]) / d_{ij}
    # softmin gradient contribution from pair (i,j): weights_pair * (1/beta) * d(d_{ij})/d(pts)
    # Wait, let me redo. softmin = -logsumexp(-beta*vals)/beta
    # d(softmin)/d(pts) = sum_j weights_j * d(val_j)/d(pts)
    # where weights_j = exp(-beta*val_j)/sum(exp(-beta*val_k))
    
    # For pair (i,j), val = d_{ij}
    # d(d_{ij})/d(pts[i]) = (pts[i] - pts[j]) / d_{ij}
    # d(d_{ij})/d(pts[j]) = (pts[j] - pts[i]) / d_{ij}
    
    # Reconstruct pair weights matrix
    pair_weights = np.zeros((N, N))
    # pair_vals was d[d < np.inf], which is row-major
    # Let's rebuild: for each i, j != i, the weight is at position in pair_vals
    # Actually, let's just compute it directly
    # The order of pair_vals: for i in range(N): for j in range(N): if j != i: d[i,j]
    # So we can build a (N,N) weight matrix
    pair_weight_matrix = np.zeros((N, N))
    # Fill from weights
    pos = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                pair_weight_matrix[i, j] = weights[pos]
                pos += 1
    
    # Now compute gradient from pairs
    # For each pair (i,j): contribution to grad[i] += pair_weight_matrix[i,j] * (pts[i]-pts[j])/d_{ij}
    # contribution to grad[j] += pair_weight_matrix[i,j] * (pts[j]-pts[i])/d_{ij}
    
    # Vectorized:
    # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
    # d_mat = cdist(pts, pts)  # (N, N)
    # Set diagonal to 1 to avoid div by zero
    d_safe = d.copy()
    d_safe[idx, idx] = 1.0
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
    # grad_pair[i] = sum_j pair_weight_matrix[i,j] * diff[i,j] / d_safe[i,j]
    # = sum_j pair_weight_matrix[i,j] * diff[i,j] / d_safe[i,j]
    # This is like a weighted sum of diff rows
    # grad_pair = (pair_weight_matrix / d_safe) @ ... no, it's element-wise
    # Let W[i,j] = pair_weight_matrix[i,j] / d_safe[i,j]
    W = pair_weight_matrix / d_safe
    W[idx, idx] = 0.0
    # grad_pair[i] = sum_j W[i,j] * diff[i,j]
    # diff[i,j] = pts[i] - pts[j]
    # sum_j W[i,j] * (pts[i] - pts[j]) = pts[i] * sum_j W[i,j] - sum_j W[i,j] * pts[j]
    # = pts[i] * W[i,:].sum() - W[i,:] @ pts
    row_sums = W.sum(axis=1)  # (N,)
    grad_pair = pts * row_sums[:, np.newaxis] - W @ pts  # (N, 2)
    
    grad += grad_pair
    
    return -softmin_val, -grad  # negative because we want to maximize softmin, minimize -softmin

def construct():
    start_time = time.time()
    
    best_pts = None
    best_radius = 0.0
    
    # Initial grid as fallback
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    best_pts = grid.copy()
    best_radius = _compute_radius(best_pts)
    
    rng = np.random.default_rng(42)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Phase 1: Multiple restarts with soft-min optimization
    n_restarts = 12
    for restart in range(n_restarts):
        if time_left() < 8.0:
            break
        
        # Random initialization
        pts = rng.uniform(-HALF + 0.01, HALF - 0.01, size=(N, 2))
        
        # Soft-min optimization with increasing beta
        beta_schedule = [10.0, 30.0, 100.0, 300.0]
        for beta in beta_schedule:
            if time_left() < 5.0:
                break
            try:
                res = minimize(
                    _soft_objective,
                    pts,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.fun < 0:
                    pts = res.x
            except Exception:
                pass
        
        # Phase 2: Exact refinement
        if time_left() < 3.0:
            break
        
        # Compute current radius
        r = _compute_radius(pts)
        
        # Exact L-BFGS-B on the exact objective (nonsmooth but works)
        def exact_obj(x):
            x2 = x.reshape(N, 2)
            r = _compute_radius(x2)
            # Gradient via finite differences is too slow; use subgradient
            # Actually, L-BFGS-B without gradient (numerical) is slow for 100 vars
            # Let's use a simple approach: just return -r and let it use numerical grad
            return -r
        
        # Use numerical gradient (L-BFGS-B default) – might be slow but we have time
        # Alternatively, skip exact refinement and just keep soft-min result
        # Let's try a few iterations of exact with numerical gradient
        if time_left() > 5.0:
            try:
                res2 = minimize(
                    exact_obj,
                    pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-14}
                )
                if res2.success:
                    pts2 = res2.x.reshape(N, 2)
                    r2 = _compute_radius(pts2)
                    if r2 > r:
                        pts = pts2
                        r = r2
            except Exception:
                pass
        
        # Update best
        if r > best_radius:
            best_radius = r
            best_pts = pts.copy()
    
    # Final check: ensure validity
    if best_pts is not None:
        # Clip to be safely inside
        # Don't clip, as it might reduce radius; just verify
        if not np.all(np.isfinite(best_pts)):
            best_pts = grid
            best_radius = _compute_radius(best_pts)
    
    return best_pts
