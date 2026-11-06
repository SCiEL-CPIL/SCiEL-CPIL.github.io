"""
Packing 200 equal circles in a unit square.
Combines hexagonal initialization, L-BFGS-B optimization with soft-min objective,
and bottleneck-driven polishing.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0

    # Distance to walls: min_i min_k (0.5 - |c_ik|)
    dw = np.minimum(np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
                    np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1]))
    wall_term = np.min(dw)

    # Pairwise distances
    n = len(centers)
    if n < 2:
        return float(wall_term)

    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0

    return min(float(wall_term), float(pair_term))


def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    # Using a lookup table or formula. For n=200, a simple loop is fast enough if called rarely,
    # but we can optimize.
    # k = i * (2n - i - 1) // 2 + j - i - 1
    # Inverting is tricky. Let's use a precomputed map or a fast binary search.
    # For simplicity and speed in the polish phase, we can just use the formula if we had it,
    # or stick to the loop which is O(N) per call. N=200 is small.
    start = 0
    for i in range(n - 1):
        block_size = n - i - 1
        if k < start + block_size:
            j = i + 1 + (k - start)
            return i, j
        start += block_size
    return 0, 1  # Fallback


def _hexagonal_grid(n_circles):
    """
    Create a hexagonal packing arrangement for n_circles circles.
    """
    best_centers = None
    best_r = 0.0
    
    # Try different numbers of rows
    for m in range(10, 25):
        base_per_row = n_circles // m
        extra = n_circles % m
        
        row_counts = []
        for i in range(m):
            if i < extra:
                row_counts.append(base_per_row + 1)
            else:
                row_counts.append(base_per_row)
        
        max_row_count = max(row_counts)
        
        r_width = 1.0 / (2.0 * max_row_count)
        r_height = 1.0 / ((m - 1) * np.sqrt(3) + 2)
        r_est = min(r_width, r_height)
        
        if r_est <= best_r and best_centers is not None:
            continue
        
        dx = 2.0 * r_est
        dy = np.sqrt(3) * r_est
        
        centers = []
        y_offset = -0.5 + r_est
        
        for row_idx, row_count in enumerate(row_counts):
            x_offset = 0.0
            if row_idx % 2 == 1:
                x_offset = dx / 2.0
            
            if row_count > 0:
                row_width = (row_count - 1) * dx
                start_x = -row_width / 2.0 + x_offset
                
            for col_idx in range(row_count):
                x = start_x + col_idx * dx
                y = y_offset + row_idx * dy
                centers.append([x, y])
        
        centers = np.array(centers)
        centers[:, 0] -= np.mean(centers[:, 0])
        centers[:, 1] -= np.mean(centers[:, 1])
        
        centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers if best_centers is not None else np.random.uniform(-0.5, 0.5, (n_circles, 2))


def _soft_min_objective_and_grad(x, t):
    """
    Compute the soft-min objective and its gradient.
    Objective: -soft_min(distances)
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Wall distances
    dx_w = 0.5 - np.abs(pts[:, 0])
    dy_w = 0.5 - np.abs(pts[:, 1])
    wall_dists = np.concatenate([dx_w, dy_w])
    
    # Pairwise distances / 2
    # Compute full distance matrix for gradient calculation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_full = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist_full, 1.0) # Avoid division by zero
    
    # We need the upper triangle for the distance list to match pdist or similar,
    # but for the sum, we can just use all off-diagonal.
    # To be consistent, let's construct the list of distances and their gradients.
    
    # It is more efficient to compute the sum of exponentials and the weighted sum of gradients
    # directly without constructing the full list of distances if possible,
    # but for N=200, the list is ~20k items, which is fine.
    
    # Get indices of upper triangle
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_full[iu] / 2.0
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Soft min calculation
    # To avoid overflow in exp, subtract max
    m = np.max(all_dists)
    exp_vals = np.exp(-(all_dists - m) / t)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    soft_min_val = m - t * np.log(sum_exp)
    obj = -soft_min_val
    
    # Gradient calculation
    grad = np.zeros_like(pts)
    
    # 1. Wall gradients
    # Weights for walls are the first 2*N weights
    w_wall_x = weights[:n]
    w_wall_y = weights[n:2*n]
    
    # d(0.5 - |x|)/dx = -sign(x)
    grad[:, 0] += w_wall_x * (-np.sign(pts[:, 0]))
    grad[:, 1] += w_wall_y * (-np.sign(pts[:, 1]))
    
    # 2. Pair gradients
    # Weights for pairs are the remaining weights
    w_pairs = weights[2*n:]
    
    # For each pair (i, j), the gradient contribution to point i is:
    # w_pair * (pts_i - pts_j) / (2 * dist_ij)
    # And to point j is:
    # w_pair * (pts_j - pts_i) / (2 * dist_ij)
    
    # We can vectorize this by scattering the weights.
    # Create a matrix of weights, initially zeros
    w_matrix = np.zeros((n, n))
    w_matrix[iu] = w_pairs
    
    # Compute term = diff / (2 * dist_full)
    # diff is (n, n, 2)
    # dist_full is (n, n)
    term = diff / (2 * dist_full[:, :, np.newaxis])
    
    # grad_pair[i] = sum_j w_matrix[i, j] * term[i, j]
    grad_pair = np.sum(w_matrix[:, :, np.newaxis] * term, axis=1)
    
    grad += grad_pair
    
    # The objective is -soft_min, so gradient is -grad(soft_min)
    return obj, -grad


def _optimize_l_bfgs(centers, max_time=60.0):
    """
    Optimize using L-BFGS-B with annealed soft-min temperature.
    """
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _compute_radius(centers)
    best_x = x0.copy()
    
    # Annealing schedule
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    start_time = time.time()
    
    for t in temperatures:
        if time.time() - start_time > max_time:
            break
            
        def objective(x):
            val, _ = _soft_min_objective_and_grad(x, t)
            return val
        
        def gradient(x):
            _, grad = _soft_min_objective_and_grad(x, t)
            return grad
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.x is not None:
                new_r = _compute_radius(res.x.reshape(-1, 2))
                if new_r > best_r:
                    best_r = new_r
                    best_x = res.x.copy()
            
            x0 = res.x.copy()
            
        except Exception as e:
            print(f"Optimization error: {e}")
            break
            
    return best_x.reshape(-1, 2), best_r


def _refine_bottleneck(centers, max_iters=50000):
    """
    Refine the packing using a repulsion-based optimizer.
    """
    centers = centers.copy()
    r = _compute_radius(centers)
    best_r = r
    best_centers = centers.copy()
    
    step_size = 0.01
    no_improve = 0
    
    for it in range(max_iters):
        if _remaining_time() < 5.0:
            break
        
        # Compute distances
        d = pdist(centers)
        min_pair_idx = np.argmin(d)
        min_pair_dist = d[min_pair_idx]
        pair_term = min_pair_dist / 2.0
        
        dw = np.minimum(np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
                        np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1]))
        min_wall_idx = np.argmin(dw)
        min_wall_dist = dw[min_wall_idx]
        wall_term = min_wall_dist
        
        current_r = min(pair_term, wall_term)
        
        if pair_term <= wall_term:
            # Pair is the bottleneck
            i, j = _pdist_index_to_pair(min_pair_idx, N)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec = np.random.randn(2)
                norm = np.linalg.norm(vec)
            direction = vec / norm
            move = step_size * direction
            centers[i] += move
            centers[j] -= move
        else:
            # Wall is the bottleneck
            idx = min_wall_idx
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            min_d = min(dl, dr, db, dt)
            if min_d == dl:
                move = np.array([step_size, 0])
            elif min_d == dr:
                move = np.array([-step_size, 0])
            elif min_d == db:
                move = np.array([0, step_size])
            else:
                move = np.array([0, -step_size])
            centers[idx] += move
        
        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
        new_r = _compute_radius(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 100:
                # Random perturbation
                idx = np.random.randint(N)
                centers[idx] += np.random.randn(2) * 0.01
                centers[idx] = np.clip(centers[idx], -0.5, 0.5)
                no_improve = 0
                step_size = max(0.001, step_size * 0.9)
            elif no_improve > 50:
                step_size = max(0.001, step_size * 0.95)
    
    return best_centers, best_r


def construct():
    """Construct optimal circle packing."""
    # Start with hexagonal grid
    centers = _hexagonal_grid(N)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Refine with L-BFGS-B
    if _remaining_time() > 20.0:
        try:
            optimized, opt_radius = _optimize_l_bfgs(centers, max_time=_remaining_time() - 10.0)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = optimized
        except Exception:
            pass
    
    # Final polish with bottleneck-driven method
    if _remaining_time() > 5.0:
        try:
            polished, pol_radius = _refine_bottleneck(best_centers)
            if pol_radius > best_radius:
                best_radius = pol_radius
                best_centers = polished
        except Exception:
            pass
    
    # Ensure all centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
