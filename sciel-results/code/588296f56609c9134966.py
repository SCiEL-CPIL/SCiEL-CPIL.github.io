"""Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
DEADLINE = 140.0  # safety margin before 150s kill
START = time.time()


def _elapsed():
    return time.time() - START


def _radius(pts):
    """Compute the common radius for a set of centres."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Wall constraint: min distance from any centre to the boundary
    # Square is [-0.5, 0.5]^2, so distance to nearest wall for point (x,y) is min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    wall_min = np.minimum(np.min(dx), np.min(dy))
    if wall_min < 0:
        return 0.0
    # Pairwise distances
    # Use a chunked approach to avoid huge memory for 200 points (actually 200 is small)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair_min = np.min(dist)
    return min(wall_min, pair_min / 2.0)


def _energy(pts, r_target):
    """
    Smooth surrogate objective: penalize violations of the constraint that all pairwise
    distances >= 2*r and all wall distances >= r.
    We want to maximize r, so we minimize a function that is 0 if feasible and positive otherwise.
    Actually, a better approach for optimization is to fix a target radius r and try to find
    a configuration where all circles of radius r fit. If we can't, we reduce r.
    
    Alternatively, we can use an objective that directly maximizes the minimum radius.
    Let's use a soft-min approach:
    f(pts) = - (soft_min of all pairwise distances/2 and wall distances)
    soft_min(x) = -log(sum(exp(-x_i / t))) * t
    
    But scipy minimize minimizes. So we define:
    objective = -soft_min(distances)
    
    However, the landscape is very rugged. A more robust method is:
    1. Start with a good initial guess (e.g., grid or hexagonal).
    2. Use a local optimizer to maximize the minimum distance.
    3. Use random restarts to escape local minima.
    
    Let's implement a local optimizer that maximizes the minimum radius using L-BFGS-B or Nelder-Mead.
    Since the objective is non-smooth (min of many functions), Nelder-Mead is often more robust.
    But for 400 dimensions, Nelder-Mead is slow.
    
    A better approach: use a smooth approximation of the minimum.
    soft_min(x, t) = -t * log(sum(exp(-x_i / t)))
    As t -> 0, soft_min -> min(x_i).
    
    We can optimize: maximize soft_min(wall_dists, pair_dists/2)
    i.e., minimize -soft_min(...)
    
    Let's implement this.
    """
    # Wall distances
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    wall_dists = np.concatenate([dx, dy])
    
    # Pairwise distances / 2
    # For 200 points, 200*200/2 = 19900 pairs. This is manageable.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    # We only need upper triangle to save memory, but for simplicity let's take all and ignore diagonal
    # Actually, let's just take all off-diagonal
    mask = ~np.eye(N, dtype=bool)
    pair_dists = dist[mask] / 2.0
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Soft min with temperature t
    t = 0.01
    # To avoid overflow, subtract max
    m = np.max(all_dists)
    log_sum = np.log(np.sum(np.exp(-(all_dists - m) / t)))
    soft_min_val = m - t * log_sum
    
    return -soft_min_val


def _optimize_local(pts0, max_iter=500):
    """
    Optimize a local configuration using L-BFGS-B on the soft-min objective.
    """
    def objective(x):
        pts = x.reshape(-1, 2)
        return _energy(pts, None)
    
    def gradient(x):
        pts = x.reshape(-1, 2)
        # Numerical gradient is slow, let's use finite differences if analytic is too complex
        # Or just use a method that doesn't need gradient like Nelder-Mead, but it's slow in high D.
        # Let's try to compute analytic gradient of soft-min.
        # soft_min = m - t * log(sum(exp(-(d_i - m)/t)))
        # d(soft_min)/d(pts) = - t * (1/sum) * sum(exp(-(d_i - m)/t) * (-1/t * d(d_i)/d(pts)))
        #                     = (1/sum) * sum(exp(-(d_i - m)/t) * d(d_i)/d(pts))
        # This is a weighted average of the gradients of the individual distances.
        
        dx = 0.5 - np.abs(pts[:, 0])
        dy = 0.5 - np.abs(pts[:, 1])
        wall_dists = np.concatenate([dx, dy])
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        mask = ~np.eye(N, dtype=bool)
        pair_dists = dist[mask] / 2.0
        
        all_dists = np.concatenate([wall_dists, pair_dists])
        t = 0.01
        m = np.max(all_dists)
        exp_vals = np.exp(-(all_dists - m) / t)
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(pts)
        
        # Wall gradients
        # d(0.5 - |x|)/dx = -sign(x)
        # d(0.5 - |y|)/dy = -sign(y)
        gx_wall = -np.sign(pts[:, 0])
        gy_wall = -np.sign(pts[:, 1])
        
        # Accumulate wall contributions
        grad[:, 0] += weights[:N] * gx_wall
        grad[:, 1] += weights[N:2*N] * gy_wall
        
        # Pairwise gradients
        # d(dist_ij / 2) / d(pts_i) = (pts_i - pts_j) / (2 * dist_ij)
        # We need to sum over j for each i.
        # This is a bit tricky with the flattened array.
        # Let's compute the gradient matrix directly.
        # grad_pair[i] = sum_{j != i} weights_pair[i,j] * (pts_i - pts_j) / (2 * dist_ij)
        
        # Reconstruct the full distance matrix for gradient computation
        dist_full = dist.copy()
        np.fill_diagonal(dist_full, 1.0)  # avoid division by zero
        
        # weights_full[i,j] = weights for pair (i,j)
        # The flattened pair_dists correspond to the upper triangle or all off-diagonal?
        # In _energy, we used mask = ~eye, which gives all off-diagonal in row-major order.
        # So pair_dists[k] corresponds to dist_full[i,j] where (i,j) is the k-th off-diagonal element.
        
        # Let's create a weight matrix
        weights_full = np.zeros((N, N))
        weights_full[mask] = weights[2*N:]  # The pair weights are the second half
        
        # Gradient for each point i:
        # grad[i] = sum_j weights_full[i,j] * (pts_i - pts_j) / (2 * dist_full[i,j])
        # Note: dist_full is symmetric, and weights_full is symmetric.
        
        # Vectorized computation:
        # diff_full = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # term = diff_full / (2 * dist_full[:, :, np.newaxis])
        # grad_pair = np.sum(weights_full[:, :, np.newaxis] * term, axis=1)
        
        term = diff / (2 * dist_full[:, :, np.newaxis])
        grad_pair = np.sum(weights_full[:, :, np.newaxis] * term, axis=1)
        
        grad += grad_pair
        
        # The objective is -soft_min, so the gradient of the objective is -grad(soft_min)
        return -grad
    
    x0 = pts0.flatten()
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    return res.x.reshape(-1, 2)


def _hexagonal_grid(n):
    """Generate a hexagonal grid of n points."""
    # This is a heuristic to generate a good initial guess.
    # For now, let's just use a square grid with a small random perturbation.
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def construct():
    best_pts = None
    best_r = 0.0
    
    # Initial guess: square grid
    pts0 = _hexagonal_grid(N)
    
    # Optimize the initial guess
    pts_opt = _optimize_local(pts0, max_iter=200)
    r = _radius(pts_opt)
    if r > best_r:
        best_r = r
        best_pts = pts_opt.copy()
    
    # Random restarts
    np.random.seed(42)
    max_restarts = 20
    for i in range(max_restarts):
        if _elapsed() > DEADLINE - 5:
            break
        
        # Generate a random initial guess
        # Start with a grid and add random noise
        pts_rand = _hexagonal_grid(N) + np.random.randn(N, 2) * 0.01
        # Clip to square
        pts_rand[:, 0] = np.clip(pts_rand[:, 0], -0.5, 0.5)
        pts_rand[:, 1] = np.clip(pts_rand[:, 1], -0.5, 0.5)
        
        # Optimize
        pts_opt = _optimize_local(pts_rand, max_iter=100)
        r = _radius(pts_opt)
        if r > best_r:
            best_r = r
            best_pts = pts_opt.copy()
    
    # Final polish: try a few more iterations on the best solution
    if best_pts is not None and _elapsed() < DEADLINE - 2:
        pts_polish = _optimize_local(best_pts, max_iter=100)
        r = _radius(pts_polish)
        if r > best_r:
            best_r = r
            best_pts = pts_polish.copy()
    
    return best_pts


# Run construct at import time to ensure it's ready? No, the evaluator will call it.
# But we need to make sure the function is defined.
# The task says: "Interface: construct() -> array of shape (N, 2)"
# So we just define the function.
