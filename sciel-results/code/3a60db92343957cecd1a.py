"""Packing 200 equal circles in a unit square with hexagonal init + repulsive relaxation."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """
    Compute the score (largest common radius) for a set of centers in [-0.5, 0.5]^2.
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0

    # Distance to walls
    min_wall = np.min(0.5 - np.abs(centers))

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    return float(min(min_wall, min_pair))


def _hex_init():
    """Generate a hexagonal lattice initialization clipped to N points in [-0.5, 0.5]^2."""
    # Hexagonal lattice with spacing s
    # We want ~200 points in a unit square
    # Hex packing density: each point occupies area s^2 * sqrt(3)/2
    # Total area = 1, so s^2 * sqrt(3)/2 * N ≈ 1 => s ≈ sqrt(2/(N*sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate hexagonal lattice points
    # Rows are offset
    dy = s * np.sqrt(3.0) / 2.0
    dx = s
    
    points = []
    # Start from -0.5 and go to 0.5
    y_start = -0.5
    y_end = 0.5
    
    row = 0
    y = y_start
    while y <= y_end + 1e-9:
        x_offset = (dx / 2.0) if (row % 2 == 1) else 0.0
        x = -0.5 + x_offset
        while x <= 0.5 + 1e-9:
            points.append([x, y])
            x += dx
        y += dy
        row += 1
    
    points = np.array(points)
    
    # Filter to those inside [-0.5, 0.5]^2
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than N, take the first N (or random sample)
    if len(points) > N:
        idx = np.random.RandomState(42).choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Need to add more points - use a finer grid
        # Fall back to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
    
    return points[:N]


def _repulsive_relaxation(centers, max_iters=5000, time_limit=None, start_time=None):
    """
    Repulsive force-based relaxation to maximize minimum distance.
    Uses soft-min surrogate for smoother gradients.
    """
    centers = centers.copy().astype(np.float64)
    N_pts = len(centers)
    
    # Soft-min parameter - smaller is sharper
    beta = 50.0  # Will adapt
    
    def soft_min_dists(centers):
        """Compute soft minimum of pairwise distances and wall distances."""
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists2, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)  # shape (N, 2)
        wall_min_per_point = np.min(wall_dists, axis=1)  # shape (N,)
        
        # For soft-min, we work with squared distances for efficiency
        # But let's work with actual distances for clarity
        dists = np.sqrt(dists2)
        np.fill_diagonal(dists, np.inf)
        
        # All distances: pairwise + wall (each wall distance counts as distance to boundary)
        # We want the minimum of all pairwise/2 and wall distances
        # Let's combine: min_pair = min(dists)/2, min_wall = min(wall_min_per_point)
        
        # Soft min over all pairwise distances and wall distances
        # Create a vector of all relevant distances
        pair_dists = dists[np.triu_indices(N_pts, k=1)] / 2.0
        wall_d = wall_min_per_point
        
        all_dists = np.concatenate([pair_dists, wall_d])
        
        # Soft min using log-sum-exp trick (for min, use -log-sum-exp of -x)
        # soft_min = -log(sum(exp(-beta * x))) / beta
        neg_beta_x = -beta * all_dists
        # Shift for numerical stability
        max_val = np.max(neg_beta_x)
        soft_min = -(max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))) / beta
        
        return soft_min
    
    def neg_soft_score(centers):
        return -soft_min_dists(centers)
    
    # Gradient of soft min w.r.t. centers
    def soft_min_grad(centers):
        N_pts = len(centers)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists2, np.inf)
        dists = np.sqrt(dists2)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        wall_min_idx = np.argmin(wall_dists, axis=1)
        wall_min_vals = np.min(wall_dists, axis=1)
        
        # Soft min over all distances
        # Pairwise: d_ij = ||c_i - c_j||/2
        # Wall: d_i = wall_min_vals[i]
        
        # We need the gradient of soft_min = -log(sum(exp(-beta * x_k))) / beta
        # where x_k are all the distances
        # d(soft_min)/d(c) = sum_k [(-1/beta) * exp(-beta * x_k) / sum(exp(-beta * x_k))] * dx_k/dc
        
        # Build weights
        pair_dists = dists[np.triu_indices(N_pts, k=1)] / 2.0
        wall_d = wall_min_vals
        
        all_dists = np.concatenate([pair_dists, wall_d])
        neg_beta_x = -beta * all_dists
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp  # These are the weights for each distance
        
        # Gradient accumulation
        grad = np.zeros_like(centers)
        
        # Pairwise contributions
        n_pairs = len(pair_dists)
        for idx in range(n_pairs):
            i, j = np.unravel_index(idx, (N_pts, N_pts))[0], np.unravel_index(idx, (N_pts, N_pts))[1]
            # Actually need to get the correct indices
            # Let's use a different approach
            pass
        
        # More efficient: use broadcasting
        # For pairwise: x_ij = ||c_i - c_j||/2, dx_ij/dc_i = (c_i - c_j)/||c_i - c_j||
        # weight_ij = weights[pair_idx]
        # contribution to grad[i] += weight_ij * (c_i - c_j)/||c_i - c_j|| / beta
        # contribution to grad[j] -= weight_ij * (c_i - c_j)/||c_i - c_j|| / beta
        
        # Let's redo this more carefully
        grad = np.zeros_like(centers)
        
        # Pairwise
        pair_i, pair_j = np.triu_indices(N_pts, k=1)
        for idx in range(len(pair_i)):
            i, j = pair_i[idx], pair_j[idx]
            w = weights[idx]
            d = dists[i, j]
            if d < 1e-12:
                continue
            # d(x_ij)/d(c_i) = (c_i - c_j)/d
            dir_ij = (centers[i] - centers[j]) / d
            grad[i] += w * dir_ij / beta
            grad[j] -= w * dir_ij / beta
        
        # Wall contributions
        # For each point i, wall distance d_i = min over k of (0.5 - |c_ik|)
        # The active wall is the one with minimum value
        for i in range(N_pts):
            w = weights[n_pairs + i]
            k = wall_min_idx[i]
            # d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
            sign = np.sign(centers[i, k])
            if sign == 0:
                sign = 1.0
            grad[i, k] -= w * sign / beta
        
        return grad
    
    # Simpler approach: use finite differences or a direct repulsion model
    # Let's use a simpler force-based approach
    
    def repulsive_forces(centers):
        """Compute repulsive forces between all pairs and from walls."""
        N_pts = len(centers)
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists2, 1e-12)
        dists = np.sqrt(dists2)
        
        # Force magnitude: 1/d^2 for close pairs
        # Only apply for pairs closer than some threshold
        threshold = 0.1
        mask = dists < threshold
        np.fill_diagonal(mask, False)
        
        force_mag = np.zeros_like(dists)
        force_mag[mask] = 1.0 / (dists[mask] ** 2 + 1e-12)
        
        # Force direction: normalized diff
        dir = diff / dists[:, :, np.newaxis]
        np.fill_diagonal(dir[:, :, 0], 0)
        np.fill_diagonal(dir[:, :, 1], 0)
        
        # Sum forces from each other point
        forces += np.sum(force_mag[:, :, np.newaxis] * dir, axis=1)
        
        # Wall repulsion: push away from walls if close
        wall_margin = 0.05
        for axis in range(2):
            # Left wall
            close_left = centers[:, axis] < -0.5 + wall_margin
            forces[close_left, axis] += (1.0 - (centers[close_left, axis] + 0.5) / wall_margin) ** 2
            # Right wall
            close_right = centers[:, axis] > 0.5 - wall_margin
            forces[close_right, axis] -= (1.0 - (0.5 - centers[close_right, axis]) / wall_margin) ** 2
        
        return forces
    
    # Adaptive step size relaxation
    step_size = 0.01
    best_score = _compute_score(centers)
    best_centers = centers.copy()
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        forces = repulsive_forces(centers)
        
        # Clamp step size
        max_force = np.max(np.abs(forces))
        if max_force < 1e-12:
            break
        
        # Normalize forces and scale
        new_centers = centers + step_size * forces / (max_force + 1e-12)
        
        # Project into bounds
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        # Check score
        score = _compute_score(new_centers)
        
        if score > best_score:
            best_score = score
            best_centers = new_centers.copy()
            centers = new_centers.copy()
            # Decrease step size slowly
            step_size *= 0.999
        else:
            # Reject step, reduce step size
            step_size *= 0.9
            if step_size < 1e-10:
                break
    
    return best_centers, best_score


def _local_optimize(centers, max_iter=1000, time_limit=None, start_time=None):
    """Local optimization using gradient ascent on soft-min objective."""
    centers = centers.copy().astype(np.float64)
    N_pts = len(centers)
    
    def soft_min_value(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists2, np.inf)
        dists = np.sqrt(dists2)
        np.fill_diagonal(dists, np.inf)
        
        pair_dists = dists[np.triu_indices(N_pts, k=1)] / 2.0
        wall_dists = 0.5 - np.abs(c)
        wall_min = np.min(wall_dists, axis=1)
        
        all_dists = np.concatenate([pair_dists, wall_min])
        return np.min(all_dists)
    
    def soft_min_grad(c, beta=100.0):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists2, np.inf)
        dists = np.sqrt(dists2)
        np.fill_diagonal(dists, np.inf)
        
        pair_dists_full = dists / 2.0
        np.fill_diagonal(pair_dists_full, np.inf)
        
        wall_dists = 0.5 - np.abs(c)
        wall_min_idx = np.argmin(wall_dists, axis=1)
        wall_min = np.min(wall_dists, axis=1)
        
        # Get upper triangle pairwise distances
        i_idx, j_idx = np.triu_indices(N_pts, k=1)
        pair_dists = pair_dists_full[i_idx, j_idx]
        
        all_dists = np.concatenate([pair_dists, wall_min])
        
        # Soft min weights
        neg_beta_x = -beta * all_dists
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        grad = np.zeros_like(c)
        
        # Pairwise
        for idx in range(len(i_idx)):
            i, j = i_idx[idx], j_idx[idx]
            w = weights[idx]
            d = dists[i, j]
            if d < 1e-12:
                continue
            dir_ij = (c[i] - c[j]) / d
            grad[i] += w * dir_ij / beta
            grad[j] -= w * dir_ij / beta
        
        # Wall
        n_pairs = len(i_idx)
        for i in range(N_pts):
            w = weights[n_pairs + i]
            k = wall_min_idx[i]
            sign = np.sign(c[i, k])
            if sign == 0:
                sign = 1.0
            grad[i, k] -= w * sign / beta
        
        return grad
    
    # Gradient ascent with adaptive step size
    lr = 0.005
    best_score = soft_min_value(centers)
    best_centers = centers.copy()
    prev_score = best_score
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        grad = soft_min_grad(centers)
        
        new_centers = centers + lr * grad
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_score = soft_min_value(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            centers = new_centers.copy()
            if new_score > prev_score:
                lr *= 1.05
            else:
                lr *= 0.95
        else:
            lr *= 0.8
            if lr < 1e-12:
                break
        
        prev_score = new_score
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Start with hexagonal initialization
    centers = _hex_init()
    score = _compute_score(centers)
    
    # Check time
    if time.time() - start_time > deadline:
        return centers
    
    # Phase 1: Repulsive relaxation
    centers_relaxed, score_relaxed = _repulsive_relaxation(
        centers, 
        max_iters=3000, 
        time_limit=60.0, 
        start_time=start_time
    )
    
    if score_relaxed > score:
        centers = centers_relaxed
        score = score_relaxed
    
    # Check time
    if time.time() - start_time > deadline - 20:
        return centers
    
    # Phase 2: Local optimization with gradient ascent on soft-min
    centers_opt, score_opt = _local_optimize(
        centers, 
        max_iter=500, 
        time_limit=40.0, 
        start_time=start_time
    )
    
    if score_opt > score:
        centers = centers_opt
        score = score_opt
    
    # Check time
    if time.time() - start_time > deadline - 10:
        return centers
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < deadline - 5:
        centers_polish, score_polish = _local_optimize(
            centers, 
            max_iter=1000, 
            time_limit=10.0, 
            start_time=start_time
        )
        if score_polish > score:
            centers = centers_polish
            score = score_polish
    
    return centers

# EVOLVE-BLOCK-END
