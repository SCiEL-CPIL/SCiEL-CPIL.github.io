"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 100

# EVOLVE-BLOCK-START

def _compute_radius_and_grad(c, beta):
    """
    Computes the soft-min radius and its gradient.
    c: (N, 3) array of centers.
    beta: inverse temperature for soft-min.
    Returns:
        r_soft: float, the soft-min radius value.
        grad: (N, 3) array, the gradient of r_soft with respect to c.
    """
    N = len(c)
    # Pairwise distances
    # dist[i, j] = ||c_i - c_j||
    # We need min_{i<j} dist[i,j] / 2
    # And min_i min_k (1/2 - |c_ik|)
    
    # Efficiently compute pairwise squared distances
    # D2[i, j] = ||c_i - c_j||^2
    # Using broadcasting
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dist = np.sqrt(dist_sq)
    dist = dist[mask]
    
    # Boundary distances
    # dist_boundary[i, k] = 1/2 - |c_ik|
    # c is in [-0.5, 0.5]
    # We want min_k (0.5 - |c_ik|)
    # This is equivalent to min_k (0.5 - abs(c_ik))
    # Let's compute the vector of boundary distances for all points and axes
    # b_dist shape (N, 3)
    b_dist = 0.5 - np.abs(c)
    
    # Combine all distances into one vector
    # Note: The objective is min( min_pair_dist/2, min_boundary_dist )
    # Let d_pair = dist / 2.0
    # Let d_bound = b_dist
    # We want soft_min( concat(d_pair, d_bound) )
    
    d_pair = dist / 2.0
    d_bound = b_dist.flatten()
    
    # All distances
    d_all = np.concatenate([d_pair, d_bound])
    
    # Soft min: -1/beta * log( sum( exp(-beta * d) ) )
    # To avoid overflow/underflow, use log-sum-exp trick
    # max_d = np.max(d_all)
    # exp_terms = np.exp(-beta * (d_all - max_d))
    # log_sum = np.log(np.sum(exp_terms))
    # soft_min = -1/beta * (log_sum - beta * max_d) = max_d - 1/beta * log_sum
    
    # However, we also need the gradient.
    # Let S = sum_k exp(-beta * d_k)
    # f = -1/beta * log(S)
    # df/dc_i = -1/beta * (1/S) * sum_k d(exp(-beta d_k))/dc_i
    # d(exp(-beta d_k))/dc_i = -beta * exp(-beta d_k) * (dd_k/dc_i)
    # df/dc_i = -1/beta * (1/S) * sum_k (-beta exp(-beta d_k) dd_k/dc_i)
    # df/dc_i = (1/S) * sum_k exp(-beta d_k) dd_k/dc_i
    # Let w_k = exp(-beta d_k) / S
    # df/dc_i = sum_k w_k dd_k/dc_i
    
    # Compute weights
    # Scale d_all to prevent overflow
    min_d = np.min(d_all)
    # Shift by min_d so that exp(-beta*(d-min_d)) is <= 1
    shifted = d_all - min_d
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    weights = exp_terms / S
    
    # Gradient calculation
    # grad = sum_k w_k * grad(d_k)
    
    # 1. Pairwise part
    # d_k corresponds to dist[i,j]/2
    # grad_{c_i} (dist[i,j]/2) = (c_i - c_j) / dist[i,j]
    # grad_{c_j} (dist[i,j]/2) = (c_j - c_i) / dist[i,j]
    
    # We have flattened d_pair. We need to map indices back.
    # This is slow if done naively. Let's use a vectorized approach for the gradient.
    
    # Reconstruct full matrix of weights for pairs
    # W_pair[i,j] = weights[k] where k corresponds to (i,j)
    # It's easier to iterate or use sparse logic, but N=100 is small enough for some vectorization.
    
    # Let's build the gradient for pairs
    grad_pair = np.zeros_like(c)
    
    # Create a matrix of weights for the upper triangle
    # We need to map the flattened index to i, j
    # For N=100, N*(N-1)/2 = 4950 pairs.
    # We can create a meshgrid of indices
    i_idx, j_idx = np.where(mask)
    
    # weights_pair = weights[:len(d_pair)]
    w_pair = weights[:len(d_pair)]
    
    # Vectorized accumulation
    # For each pair (i, j) with weight w:
    # grad[i] += w * (c[i] - c[j]) / dist[i,j]
    # grad[j] += w * (c[j] - c[i]) / dist[i,j]
    
    # dist_matrix for pairs
    # We have dist vector corresponding to i_idx, j_idx
    # Avoid division by zero (should not happen if distinct)
    safe_dist = np.where(dist > 1e-12, dist, 1.0)
    
    # diff_vec = c[i] - c[j]
    # We can compute this vectorized
    # c_i = c[i_idx], c_j = c[j_idx]
    c_i = c[i_idx]
    c_j = c[j_idx]
    diff_vec = c_i - c_j
    
    # unit vectors
    unit_vecs = diff_vec / safe_dist[:, np.newaxis]
    
    # weighted unit vectors
    w_vecs = w_pair[:, np.newaxis] * unit_vecs
    
    # Accumulate to grad_pair
    # np.add.at
    np.add.at(grad_pair, i_idx, w_vecs)
    np.add.at(grad_pair, j_idx, -w_vecs)
    
    # 2. Boundary part
    # d_bound[i, k] = 0.5 - |c_ik|
    # grad_{c_ik} d_bound = -sign(c_ik)
    # w_bound = weights[len(d_pair):]
    w_bound = weights[len(d_pair):]
    
    # Reshape w_bound to (N, 3)
    w_bound_mat = w_bound.reshape(N, 3)
    
    # grad_bound[i, k] = w_bound[i, k] * (-sign(c[i, k]))
    # sign(0) is 0, which is fine (subgradient 0)
    signs = np.sign(c)
    grad_bound = -w_bound_mat * signs
    
    # Total gradient
    grad = grad_pair + grad_bound
    
    # Value
    # soft_min = min_d - 1/beta * log(S)
    # Wait, the shift was min_d.
    # S_shifted = sum exp(-beta (d - min_d))
    # log(S) = log(S_shifted) - beta * min_d
    # f = -1/beta (log(S_shifted) - beta min_d) = min_d - 1/beta log(S_shifted)
    r_soft = min_d - (1.0 / beta) * np.log(S)
    
    return r_soft, grad

def _exact_radius(c):
    """Compute exact max common radius for centres c."""
    n = len(c)
    if n == 0:
        return 0.0
    # Boundary clearance
    d_boundary = 0.5 - np.abs(c)
    min_boundary = np.min(d_boundary)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)

def _fcc_init(n_points=100):
    """Initialize using FCC lattice points, take best n_points."""
    # FCC basis vectors in unit cell
    a = 1.0 / 4.0  # lattice constant scaling
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ])
    # Generate FCC points in a larger region
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    # Scale to fit in [-0.5, 0.5]
                    p = p * 0.15
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    points = np.array(points)
    if len(points) < n_points:
        # Fall back to grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        np.random.seed(42)
        idx = np.random.permutation(len(grid))
        c = grid[idx[:n_points]].copy()
        c += np.random.normal(0, 0.01, c.shape)
        return np.clip(c, -0.49, 0.49)
    # Select n_points with best local density
    np.random.seed(42)
    idx = np.random.permutation(len(points))
    c = points[idx[:n_points]].copy()
    c += np.random.normal(0, 0.005, c.shape)
    return np.clip(c, -0.49, 0.49)

def _active_set_polish(c, max_time=10.0):
    """Polish using active-set: only consider pairs near the minimum."""
    start = time.time()
    deadline = start + max_time
    n = len(c)
    best_c = c.copy()
    best_r = _exact_radius(c)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        r = _exact_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Find active constraints (pairs with distance/2 close to r, and boundary close to r)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        boundary_dists = 0.5 - np.abs(c)
        
        # Active threshold: within 5% of current radius
        threshold = r * 1.05
        active_pairs = np.argwhere((pair_dists < threshold) & (pair_dists > 1e-10))
        active_boundary = np.argwhere(boundary_dists < threshold)
        
        if len(active_pairs) == 0 and len(active_boundary) == 0:
            break
        
        # Build gradient from active constraints only
        grad = np.zeros_like(c)
        
        # Pair gradients
        for i, j in active_pairs:
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-12:
                continue
            direction = (c[i] - c[j]) / d
            # We want to increase distance, so push apart
            grad[i] += direction
            grad[j] -= direction
        
        # Boundary gradients
        for i, k in active_boundary:
            # Push away from boundary
            if c[i, k] > 0:
                grad[i, k] -= 1.0
            else:
                grad[i, k] += 1.0
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Adaptive step size
        step = 0.002 * (r / (grad_norm + 1e-10))
        step = min(step, 0.01)
        
        c_new = c + step * grad
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        r_new = _exact_radius(c_new)
        if r_new >= r:
            c = c_new
        else:
            # Try smaller step
            c_new = c + 0.5 * step * grad
            c_new = np.clip(c_new, -0.4999, 0.4999)
            r_new = _exact_radius(c_new)
            if r_new >= r:
                c = c_new
            else:
                # Random perturbation
                c_new = c + np.random.normal(0, 0.002, c.shape)
                c_new = np.clip(c_new, -0.4999, 0.4999)
                r_new = _exact_radius(c_new)
                if r_new > r:
                    c = c_new
    
    return best_c

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Better initialization
    c = _fcc_init(N)
    
    best_c = c.copy()
    best_score = _exact_radius(c)
    
    # Phase 1: Soft-min gradient ascent with momentum
    momentum = np.zeros_like(c)
    beta_current = 10.0
    beta_max = 300.0
    step_size = 0.005
    momentum_coeff = 0.9
    
    phase1_deadline = start_time + 80.0  # 80s for phase 1
    
    while time.time() < phase1_deadline:
        # Gradually increase beta
        beta_current = min(beta_current * 1.02, beta_max)
        
        r, grad = _compute_radius_and_grad(c, beta_current)
        
        # Momentum update
        momentum = momentum_coeff * momentum + (1 - momentum_coeff) * grad
        momentum_norm = np.linalg.norm(momentum)
        if momentum_norm > 1e-12:
            momentum_normed = momentum / momentum_norm
        else:
            momentum_normed = momentum
        
        c_new = c + step_size * momentum_normed
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        r_new, _ = _compute_radius_and_grad(c_new, beta_current)
        
        if r_new > r:
            c = c_new
            # Occasionally check exact radius
            if int((time.time() - start_time) * 10) % 5 == 0:
                exact_r = _exact_radius(c)
                if exact_r > best_score:
                    best_score = exact_r
                    best_c = c.copy()
        else:
            step_size *= 0.95
            step_size = max(step_size, 0.0005)
    
    # Phase 2: Active-set polish
    remaining = deadline - time.time()
    if remaining > 2.0:
        c = best_c
        c = _active_set_polish(c, max_time=min(remaining - 1.0, 50.0))
        final_r = _exact_radius(c)
        if final_r > best_score:
            best_score = final_r
            best_c = c.copy()
    
    # Final validation
    best_c = np.clip(best_c, -0.499999, 0.499999)
    
    return best_c

# EVOLVE-BLOCK-END
