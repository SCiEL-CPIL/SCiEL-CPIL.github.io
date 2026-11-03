"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return float(min_dist_to_wall)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _repulsion_potential(centers, k=10):
    """Compute sum of inverse-power repulsions between all pairs and walls."""
    # Pairwise repulsion: sum over pairs of 1/d^k
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Avoid division by zero
    sq_dists = np.maximum(sq_dists, 1e-20)
    
    # Pairwise: sum of (1/d)^k = sum of 1/sq_dists^(k/2)
    pair_repulsion = np.sum(1.0 / (sq_dists ** (k / 2.0)))
    
    # Wall repulsion: sum of 1/wall_dist^k for each coordinate
    wall_dists = 0.5 - np.abs(centers)
    wall_dists = np.maximum(wall_dists, 1e-10)
    wall_repulsion = np.sum(1.0 / (wall_dists ** k))
    
    return pair_repulsion + wall_repulsion

def _repulsion_grad(centers, k=10):
    """Compute gradient of repulsion potential."""
    centers = np.asarray(centers)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    sq_dists = np.maximum(sq_dists, 1e-20)
    
    # d/dx_i of (1/d_ij^k) = -k * (x_i - x_j) / d_ij^(k+2)
    # = -k * diff / sq_dists^((k+2)/2)
    inv_dist_pow = 1.0 / (sq_dists ** ((k + 2) / 2.0))
    np.fill_diagonal(inv_dist_pow, 0.0)
    
    # grad_i = -k * sum_j diff_ij * inv_dist_pow_ij
    grad_pair = -k * np.sum(diff * inv_dist_pow[:, :, np.newaxis], axis=1)
    
    # Wall gradient
    wall_dists = 0.5 - np.abs(centers)
    wall_dists = np.maximum(wall_dists, 1e-10)
    signs = np.sign(centers)
    signs[centers == 0] = 1.0
    
    # d/dx_i of (1/wall_dist^k) = -k * sign(x_i) / wall_dist^(k+1)
    grad_wall = -k * signs / (wall_dists ** (k + 1))
    
    grad = grad_pair + grad_wall
    
    return grad

def _soft_min_radius(centers, temperature=0.01):
    """Smooth surrogate for the radius using log-sum-exp soft-min."""
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Soft-min of pairwise distances / 2
    x_pair = sq_dists / 4.0  # (d/2)^2
    finite_mask = np.isfinite(x_pair)
    if not np.any(finite_mask):
        return 0.0
    
    max_x = np.max(x_pair[finite_mask])
    exp_terms = np.zeros_like(x_pair)
    exp_terms[finite_mask] = np.exp(-(x_pair[finite_mask] - max_x) / temperature**2)
    sum_exp = np.sum(exp_terms)
    if sum_exp > 0:
        soft_min_sq_pair = max_x - temperature**2 * np.log(sum_exp)
    else:
        soft_min_sq_pair = max_x
    soft_min_pair = np.sqrt(max(soft_min_sq_pair, 0.0))
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    max_wall = np.max(wall_dists)
    exp_wall = np.exp(-(wall_dists - max_wall) / temperature**2)
    sum_wall = np.sum(exp_wall)
    if sum_wall > 0:
        soft_min_wall = max_wall - temperature**2 * np.log(sum_wall)
    else:
        soft_min_wall = max_wall
    
    # Combine: soft min of (soft_min_pair, soft_min_wall)
    vals = np.array([soft_min_pair, soft_min_wall])
    max_v = np.max(vals)
    exp_v = np.exp(-(vals - max_v) / temperature**2)
    soft_min_combined = max_v - temperature**2 * np.log(np.sum(exp_v))
    
    return float(soft_min_combined)

def _objective_and_grad(x, temperature=0.01):
    """Objective: maximize soft-min radius. Returns (-radius, gradient)."""
    centers = x.reshape(N, 3)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    T2 = temperature**2
    
    # Pairwise soft-min
    finite_mask = np.isfinite(sq_dists)
    if np.any(finite_mask):
        max_sq = np.max(sq_dists[finite_mask])
        exp_terms = np.zeros_like(sq_dists)
        exp_terms[finite_mask] = np.exp(-(sq_dists[finite_mask] - max_sq) / (4.0 * T2))
        sum_exp = np.sum(exp_terms)
        if sum_exp > 0:
            log_sum = np.log(sum_exp)
            soft_min_sq_pair = max_sq - 4.0 * T2 * log_sum
        else:
            soft_min_sq_pair = max_sq
    else:
        soft_min_sq_pair = 0.0
    
    soft_min_sq_pair = max(soft_min_sq_pair, 0.0)
    soft_min_pair = np.sqrt(soft_min_sq_pair)
    
    # Wall soft-min
    wall_dists = 0.5 - np.abs(centers)
    max_wall = np.max(wall_dists)
    exp_wall = np.exp(-(wall_dists - max_wall) / T2)
    sum_wall = np.sum(exp_wall)
    if sum_wall > 0:
        soft_min_wall = max_wall - T2 * np.log(sum_wall)
    else:
        soft_min_wall = max_wall
    
    # Combined soft-min
    vals = np.array([soft_min_pair, soft_min_wall])
    max_v = np.max(vals)
    exp_v = np.exp(-(vals - max_v) / T2)
    sum_v = np.sum(exp_v)
    soft_min_combined = max_v - T2 * np.log(sum_v)
    
    # Gradient computation
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    if np.any(finite_mask) and sum_exp > 0:
        weights = exp_terms / sum_exp  # (N, N) matrix
        sum_w = np.sum(weights, axis=1)  # (N,)
        weighted_sum_x = weights @ centers  # (N, 3)
        
        grad_pair_sq = 4.0 * (centers * sum_w[:, np.newaxis] - weighted_sum_x)
        
        if soft_min_sq_pair > 1e-12:
            grad_pair = grad_pair_sq / (2.0 * soft_min_pair)
        else:
            grad_pair = np.zeros_like(centers)
        
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        
        d_sum_wall_dx = exp_wall * signs / T2  # (N, 3)
        d_soft_min_wall_dx = -T2 * d_sum_wall_dx / sum_wall
        
        exp_pair_v = np.exp(-(soft_min_pair - max_v) / T2)
        exp_wall_v = np.exp(-(soft_min_wall - max_v) / T2)
        
        grad_combined = (exp_pair_v / sum_v) * grad_pair + (exp_wall_v / sum_v) * d_soft_min_wall_dx
        
        grad = grad_combined
    
    # We want to MAXIMIZE soft_min_combined, so minimize -soft_min_combined
    return -soft_min_combined, -grad

def _optimize_from_init(init_centers, temperature=0.01, max_iter=5000, time_limit=20.0):
    """Optimize from initial centers using L-BFGS-B."""
    start_time = time.time()
    x0 = init_centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        val, grad = _objective_and_grad(x, temperature)
        return val, grad
    
    result = minimize(
        obj, x0,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    
    return result.x.reshape(N, 3)

def _polish_critical_pairs(centers, time_limit=5.0):
    """Final high-precision polish focusing only on critical (closest) pairs."""
    start_time = time.time()
    centers = centers.copy()
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Find the minimum radius constraint
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        min_wall_dist = np.min(0.5 - np.abs(centers))
        min_wall_idx = np.unravel_index(np.argmin(0.5 - np.abs(centers)), (N, 3))
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # If pair distance is limiting, push those two apart
        if min_pair_dist / 2.0 < min_wall_dist:
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-6, 0, 0])
                dist = 1e-6
            # Push apart along the connecting line
            push = (r - dist/2.0) * d / dist * 0.5
            centers[i] += push
            centers[j] -= push
        else:
            # Wall is limiting, push that sphere inward
            i, k = min_wall_idx
            wall = 0.5 - np.abs(centers[i, k])
            if wall < r:
                sign = np.sign(centers[i, k])
                if sign == 0:
                    sign = 1.0
                centers[i, k] += (r - wall) * sign
        
        # Re-clip to cube
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _bcc_init(n):
    """Generate body-centered cubic lattice initial configuration."""
    # BCC lattice: simple cubic points + body center
    m = int(np.ceil((n / 2) ** (1.0/3.0)))
    spacing = 1.0 / m
    centers = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Simple cubic corner
                x = (ix + 0.5) * spacing - 0.5
                y = (iy + 0.5) * spacing - 0.5
                z = (iz + 0.5) * spacing - 0.5
                centers.append([x, y, z])
                # Body center (only for interior cells to avoid duplicates at boundaries)
                if ix < m-1 and iy < m-1 and iz < m-1:
                    bx = (ix + 1.0) * spacing - 0.5
                    by = (iy + 1.0) * spacing - 0.5
                    bz = (iz + 1.0) * spacing - 0.5
                    centers.append([bx, by, bz])
    
    centers = np.array(centers)
    if len(centers) > n:
        # Take the first n points
        centers = centers[:n]
    elif len(centers) < n:
        # Pad with jittered grid points
        grid = _grid_init(n)
        centers = np.vstack([centers, grid[len(centers):n]])
    
    return centers

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _jittered_grid_init(n, seed=None):
    """Generate jittered grid initial configuration."""
    grid = _grid_init(n)
    rng = np.random.RandomState(seed)
    jitter = rng.uniform(-0.02, 0.02, size=(n, 3))
    return np.clip(grid + jitter, -0.49, 0.49)

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple initializations
    inits = []
    grid = _grid_init(N)
    inits.append(('grid', grid))
    
    bcc = _bcc_init(N)
    inits.append(('bcc', bcc))
    
    for seed in range(6):
        inits.append((f'rand{seed}', _random_init(N, seed=seed)))
    
    for seed in range(4):
        inits.append((f'jitter{seed}', _jittered_grid_init(N, seed=seed)))
    
    # Evaluate all inits
    for name, init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
    
    # Stage 1: Coarse optimization with larger temperature
    temps = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    active_inits = [best_centers]
    for name, init in inits[:4]:
        active_inits.append(init)
    
    for temp in temps:
        if time.time() > deadline - 50.0:
            break
        
        for init in active_inits:
            if time.time() > deadline - 35.0:
                break
            
            try:
                time_remaining = deadline - time.time() - 25.0
                if time_remaining < 5.0:
                    break
                optimized = _optimize_from_init(init, temperature=temp, 
                                                max_iter=3000, time_limit=min(15.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
        
        # Update active inits with best found so far
        active_inits = [best_centers] + active_inits[:2]
    
    # Stage 2: Fine polishing from best with very small temperature
    if time.time() < deadline - 20.0:
        fine_temps = [0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
        for temp in fine_temps:
            if time.time() > deadline - 10.0:
                break
            
            try:
                time_remaining = deadline - time.time() - 5.0
                if time_remaining < 3.0:
                    break
                optimized = _optimize_from_init(best_centers, temperature=temp,
                                                max_iter=5000, time_limit=min(10.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Stage 3: Final critical-pair polish
    if time.time() < deadline - 5.0:
        try:
            polished = _polish_critical_pairs(best_centers, time_limit=4.0)
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished
        except Exception:
            pass
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
# EVOLVE-BLOCK-END
