import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_min = min(np.min(dx), np.min(dy))
    if wall_min < 0:
        return 0.0
    # Pairwise distances
    dists = pdist(centers)
    pair_min = np.min(dists) / 2.0
    return min(wall_min, pair_min)


def _hexagonal_grid(n):
    """Generate a hexagonal grid of n points scaled to fit in [-0.5, 0.5]^2."""
    # Approximate square root for grid dimensions
    # Hexagonal packing density is higher, so we can fit more points
    # For a square container, a rectangular hexagonal grid is a good start.
    # Let's try to fit a grid of m x m points, but with hexagonal offset.
    # Actually, just generating a grid and then optimizing is fine.
    # But a hexagonal start is better.
    
    # Estimate rows and cols
    # Area per circle in hex packing is ~ 2.31 * r^2.
    # Total area = 1.0. N=200. r ~ sqrt(1/200/2.31) ~ 0.046.
    # Spacing ~ 2r ~ 0.09.
    # Width 1.0 / 0.09 ~ 11 cols.
    # Height 1.0 / (0.09*sqrt(3)/2) ~ 11 rows.
    
    cols = 14
    rows = 15
    # Total points cols*rows = 210. We need 200.
    
    points = []
    for r in range(rows):
        for c in range(cols):
            # Hexagonal offset
            x = (c + 0.5 * (r % 2)) / cols - 0.5
            y = r / rows - 0.5
            # Scale to fit in [-0.5, 0.5] with some margin
            # The natural spacing is 1/cols.
            # Let's just generate a dense grid and take the first N, or center them.
            points.append([x, y])
            
    pts = np.array(points)
    if len(pts) > n:
        # Take a centered subset or just the first N
        # Better to take a random subset or a centered block
        # For simplicity, take first N, but shuffle to break symmetry?
        # No, keep structure.
        pts = pts[:n]
    elif len(pts) < n:
        # Pad with random
        extra = np.random.randn(n - len(pts), 2) * 0.1
        extra[:, 0] = np.clip(extra[:, 0], -0.4, 0.4)
        extra[:, 1] = np.clip(extra[:, 1], -0.4, 0.4)
        pts = np.vstack([pts, extra])
        
    return pts


def _soft_min_objective_and_grad(x, t=0.005):
    """
    Compute the negative soft-min objective and its gradient.
    x is a flat array of length 2*N.
    We want to maximize the min distance, so we minimize -soft_min.
    """
    pts = x.reshape(-1, 2)
    
    # Wall distances
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    wall_dists = np.concatenate([dx, dy])
    
    # Pairwise distances / 2
    dists = pdist(pts)
    pair_dists = dists / 2.0
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Soft min
    # To avoid overflow in exp, shift by max
    m = np.max(all_dists)
    exp_vals = np.exp(-(all_dists - m) / t)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    soft_min_val = m - t * np.log(sum_exp)
    
    # Gradient of soft_min with respect to pts
    grad = np.zeros_like(pts)
    
    # Wall gradients
    # d(0.5 - |x|)/dx = -sign(x) for x != 0
    gx_wall = -np.sign(pts[:, 0])
    gy_wall = -np.sign(pts[:, 1])
    
    grad[:, 0] += weights[:N] * gx_wall
    grad[:, 1] += weights[N:2*N] * gy_wall
    
    # Pairwise gradients
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    weights_pair = weights[2*N:2*N + num_pairs]
    
    # Compute differences
    diffs = pts[i_idx] - pts[j_idx]  # shape (num_pairs, 2)
    dists_matrix = dists  # shape (num_pairs,)
    
    # Avoid division by zero
    safe_dists = np.where(dists_matrix > 1e-10, dists_matrix, 1.0)
    
    # term_k = weights_pair[k] * diffs[k] / (2 * safe_dists[k])
    terms = weights_pair[:, np.newaxis] * diffs / (2.0 * safe_dists[:, np.newaxis])
    
    # Accumulate into grad
    np.add.at(grad, i_idx, terms)
    np.add.at(grad, j_idx, -terms)
    
    # The objective is -soft_min, so the gradient of the objective is -grad(soft_min)
    obj_val = -soft_min_val
    grad_val = -grad.flatten()
    
    return obj_val, grad_val


def _optimize_l_bfgs(x0, max_iter=100, t=0.005):
    """Optimize using L-BFGS-B on the soft-min objective."""
    
    def objective_and_grad(x):
        val, grad = _soft_min_objective_and_grad(x, t)
        return val, grad
    
    res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000})
    return res.x.reshape(-1, 2)


def _bottleneck_polish(pts, iterations=10):
    """
    Identify the pairs with the smallest distance and nudge them apart.
    """
    for _ in range(iterations):
        if _remaining_time() < 1.0:
            break
        
        # Compute all pairwise distances
        dists = pdist(pts)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        
        # Also check wall distances
        dx = 0.5 - np.abs(pts[:, 0])
        dy = 0.5 - np.abs(pts[:, 1])
        wall_dists = np.concatenate([dx, dy])
        min_wall_idx = np.argmin(wall_dists)
        min_wall_dist = wall_dists[min_wall_idx]
        
        # Find the bottleneck
        if min_dist < min_wall_dist:
            # Bottleneck is a pair
            # Convert pdist index to (i, j)
            # For N points, index k corresponds to:
            # i = floor((N - 0.5) - sqrt((N - 0.5)^2 - 4*k))
            # j = k - i*(N - 1 - i)/2 + i
            n = N
            i = int(np.floor((n - 0.5) - np.sqrt((n - 0.5)**2 - 4*min_dist_idx)))
            j = min_dist_idx - i*(n - 1 - i)//2 + i
            
            # Nudge i and j apart
            diff = pts[i] - pts[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                direction = diff / norm
                # Move i in direction, j in -direction
                step = 0.001 * (min_dist / 2.0) # Small step proportional to current gap
                pts[i] += direction * step
                pts[j] -= direction * step
        else:
            # Bottleneck is a wall
            # min_wall_idx < N means x-wall, >= N means y-wall
            if min_wall_idx < N:
                idx = min_wall_idx
                # Move away from wall
                if pts[idx, 0] > 0:
                    pts[idx, 0] -= 0.001
                else:
                    pts[idx, 0] += 0.001
            else:
                idx = min_wall_idx - N
                if pts[idx, 1] > 0:
                    pts[idx, 1] -= 0.001
                else:
                    pts[idx, 1] += 0.001
        
        # Clip to square
        pts[:, 0] = np.clip(pts[:, 0], -0.499, 0.499)
        pts[:, 1] = np.clip(pts[:, 1], -0.499, 0.499)
        
    return pts


def construct():
    best_pts = None
    best_r = 0.0
    
    np.random.seed(42)
    
    # Initial guess: hexagonal-like grid
    pts0 = _hexagonal_grid(N)
    
    # Optimize the initial guess
    pts_opt = _optimize_l_bfgs(pts0.flatten(), max_iter=200, t=0.005)
    pts_opt = _bottleneck_polish(pts_opt, iterations=20)
    r = _compute_radius(pts_opt)
    if r > best_r:
        best_r = r
        best_pts = pts_opt.copy()
    
    # Random restarts with different initial perturbations
    max_restarts = 50
    for i in range(max_restarts):
        if _remaining_time() < 10.0:
            break
        
        # Generate a random initial guess based on the grid
        noise_scale = 0.01 + 0.02 * (i % 5)
        pts_rand = _hexagonal_grid(N) + np.random.randn(N, 2) * noise_scale
        pts_rand[:, 0] = np.clip(pts_rand[:, 0], -0.49, 0.49)
        pts_rand[:, 1] = np.clip(pts_rand[:, 1], -0.49, 0.49)
        
        # Optimize
        # Use a smaller t for finer optimization
        t_val = 0.002 + 0.001 * (i % 3)
        pts_opt = _optimize_l_bfgs(pts_rand.flatten(), max_iter=100, t=t_val)
        pts_opt = _bottleneck_polish(pts_opt, iterations=10)
        r = _compute_radius(pts_opt)
        if r > best_r:
            best_r = r
            best_pts = pts_opt.copy()
            
    # Final polish on the best solution
    if best_pts is not None and _remaining_time() > 5.0:
        pts_polish = _optimize_l_bfgs(best_pts.flatten(), max_iter=100, t=0.001)
        pts_polish = _bottleneck_polish(pts_polish, iterations=50)
        r = _compute_radius(pts_polish)
        if r > best_r:
            best_r = r
            best_pts = pts_polish.copy()
    
    return best_pts
