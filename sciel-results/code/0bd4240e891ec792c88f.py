"""Binary search on radius with L-BFGS-B soft-min optimization for 50 circles in unit square."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5


def _score(c):
    """Largest common radius for centres c (N,2)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_bound = np.min(HALF - np.abs(c))
    return min(min_pair, min_bound)


def _soft_min_obj_and_grad(x, beta=100.0):
    """
    Soft-min objective for maximizing the minimum distance.
    Returns (-soft_min, gradient) for minimization.
    x is flattened (N*2,) array of centers.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]  # (N, N, 2)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)  # (N, N)
    
    # Pair scores: dist/2
    pair_scores = 0.5 * dist
    # Wall scores: HALF - |coord|
    wall_scores = HALF - np.abs(pts)  # (N, 2)
    
    # Combine all scores into one vector
    # Pair scores: N*(N-1)/2 unique pairs (use upper triangle)
    iu = np.triu_indices(N, k=1)
    pair_vals = pair_scores[iu]  # (N*(N-1)/2,)
    
    # Wall scores: N*2 values
    wall_vals = wall_scores.flatten()  # (N*2,)
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Soft min: -log(sum(exp(-beta * vals))) / beta
    # For numerical stability, subtract max
    max_val = np.max(all_vals)
    shifted = all_vals - max_val
    exp_shifted = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_shifted))
    soft_min = max_val - log_sum / beta
    
    # Gradient of soft_min w.r.t. x
    # d(soft_min)/dx = (1/beta) * sum_j exp(-beta*val_j) * d(val_j)/dx / sum_j exp(-beta*val_j)
    # = sum_j w_j * d(val_j)/dx where w_j = exp(-beta*val_j) / sum(exp(...))
    
    weights = exp_shifted / np.sum(exp_shifted)
    pair_weights = weights[:len(pair_vals)]
    wall_weights = weights[len(pair_vals):]
    
    # Gradient w.r.t. pts from pairs
    pair_grad = np.zeros((N, 2))
    # For each pair (i,j), d(dist_ij/2)/d(pts_i) = (pts_i - pts_j) / dist_ij
    # We need to accumulate: sum_j w_ij * (pts_i - pts_j) / dist_ij
    # Using the upper triangle indices
    i_idx, j_idx = iu
    for idx in range(len(pair_vals)):
        i, j = i_idx[idx], j_idx[idx]
        d = dist[i, j]
        if d > 1e-12:
            w = pair_weights[idx]
            diff_ij = pts[i] - pts[j]
            pair_grad[i] += w * diff_ij / d
            pair_grad[j] -= w * diff_ij / d
    
    # Gradient w.r.t. pts from walls
    wall_grad = np.zeros((N, 2))
    for i in range(N):
        for k in range(2):
            sign_x = np.sign(pts[i, k])
            if sign_x == 0:
                sign_x = 1.0
            # d(HALF - |x|)/dx = -sign(x)
            wall_grad[i, k] = -sign_x * wall_weights[i * 2 + k]
    
    total_grad_pts = pair_grad + wall_grad
    total_grad = total_grad_pts.flatten()
    
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min, -total_grad


def _lbfgs_optimize(pts, beta=100.0, max_iter=2000, deadline=None):
    """
    Optimize circle centers using L-BFGS-B with soft-min surrogate.
    Returns (best_pts, best_score).
    """
    x0 = pts.flatten()
    bounds = [(-HALF, HALF)] * (2 * N)
    
    best_score = _score(pts)
    best_pts = pts.copy()
    
    def obj_and_grad(x):
        if deadline is not None and time.time() > deadline:
            return 1e10, np.zeros_like(x)
        val, grad = _soft_min_obj_and_grad(x, beta=beta)
        return val, grad
    
    try:
        result = minimize(
            obj_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        new_pts = result.x.reshape(N, 2)
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_pts = new_pts
            best_score = new_score
    except Exception:
        pass
    
    return best_pts, best_score


def _hex_seed(r):
    """Generate a hexagonal lattice seed with spacing 2r."""
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    while y <= HALF - r + 1e-9:
        x = -HALF + r + (0.5 * dx if (row % 2) else 0.0)
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    pts = np.array(list(zip(xs, ys)))
    if len(pts) >= N:
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[:N - len(pts)]])
    return np.clip(pts, -HALF + r, HALF - r)


def _grid_seed(r):
    """Generate a grid seed."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def _feasible(pts, r, max_iter=500, deadline=None):
    """
    Try to find a configuration of N circles of radius r inside the square.
    Uses L-BFGS-B to maximize the soft-min distance. Returns (success, pts).
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    
    # Quick check if already feasible
    dist = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_wall = np.min(HALF - np.abs(pts))
    
    if min_pair >= 2 * r - 1e-10 and min_wall >= r - 1e-10:
        return True, pts
    
    # Use L-BFGS-B to optimize
    optimized_pts, score = _lbfgs_optimize(pts, beta=100.0, max_iter=max_iter, deadline=deadline)
    
    # Check feasibility
    dist = np.sqrt(np.sum((optimized_pts[:, None, :] - optimized_pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_wall = np.min(HALF - np.abs(optimized_pts))
    
    success = (min_pair >= 2 * r - 1e-10) and (min_wall >= r - 1e-10)
    return success, optimized_pts


def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Start with a reasonable radius estimate
    r_low = 0.06
    r_high = 0.08
    
    best_pts = _hex_seed(0.071).copy()
    best_r = _score(best_pts)
    
    # Binary search on radius
    for it in range(30):
        if time.time() > deadline:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        success = False
        trial_pts = None
        
        seeds = [_hex_seed(r_mid), _grid_seed(r_mid)]
        for _ in range(2):
            seed = _hex_seed(r_mid) + np.random.randn(N, 2) * 0.01
            seed = np.clip(seed, -HALF + r_mid, HALF + r_mid)
            seeds.append(seed)
        
        for seed in seeds:
            if time.time() > deadline:
                break
            ok, pts = _feasible(seed, r_mid, max_iter=300, deadline=deadline)
            if ok:
                success = True
                trial_pts = pts
                break
        
        if success:
            r_low = r_mid
            best_pts = trial_pts.copy()
            best_r = _score(best_pts)
        else:
            r_high = r_mid
    
    # Final high-precision refinement with L-BFGS-B
    if time.time() < deadline - 10:
        # Try multiple beta values for refinement
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 5:
                break
            polished_pts, polished_r = _lbfgs_optimize(best_pts, beta=beta, max_iter=2000, deadline=deadline - 3)
            if polished_r > best_r:
                best_pts = polished_pts
                best_r = polished_r
    
    return best_pts
