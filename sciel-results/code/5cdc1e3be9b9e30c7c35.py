"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(positions):
    """Compute the exact minimum radius."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    min_boundary = np.min(np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    ))
    
    if min_boundary <= 0:
        return 0.0

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(min_boundary, min_pair / 2.0)

def _generate_hexagonal_grid(n_circles, spacing=None):
    """Generate a hexagonal grid that fits well in the unit square."""
    if spacing is None:
        spacing = 0.073
    
    points = []
    max_extent = 1.0
    row_h = spacing * np.sqrt(3) / 2
    n_rows = int(max_extent / row_h) + 2
    
    for row in range(-n_rows, n_rows + 1):
        offset = (row % 2) * 0.5 * spacing
        x_max = int(max_extent / spacing) + 1
        for col in range(-x_max, x_max + 1):
            x = col * spacing + offset
            y = row * row_h
            if abs(x) <= 0.495 and abs(y) <= 0.495:
                points.append([x, y])
            
    points = np.array(points)
    
    if len(points) < n_circles:
        spacing = 0.068
        row_h = spacing * np.sqrt(3) / 2
        n_rows = int(max_extent / row_h) + 2
        points = []
        for row in range(-n_rows, n_rows + 1):
            offset = (row % 2) * 5 * 0.5 * spacing
            x_max = int(max_extent / spacing) + 1
            for col in range(-x_max, x_max + 1):
                x = col * spacing + offset
                y = row * row_h
                if abs(x) <= 0.495 and abs(y) <= 0.495:
                    points.append([x, y])
        points = np.array(points)
    
    # Sort by distance to center and pick the n_circles closest
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)[:n_circles]
    points = points[idx].copy()
    
    return points

def _inverse_power_energy(positions, power=6, eps=1e-8):
    """
    Compute the inverse-power-sum energy: sum of 1/d^power for all pairs.
    Lower energy means more spread out.
    Also add boundary penalty.
    """
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Inverse power sum for pairs
    energy = np.sum(1.0 / (dist_sq + eps) ** (power / 2.0))
    
    # Boundary penalty: penalize getting close to edges
    boundary_dist = np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    )
    # Strong penalty for being outside or very close to boundary
    boundary_penalty = np.sum(1.0 / (np.maximum(boundary_dist, eps) ** power))
    
    return energy + 10.0 * boundary_penalty

def _inverse_power_grad(positions, power=6, eps=1e-8):
    """
    Compute gradient of inverse-power-sum energy.
    """
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    
    # d/dx_i [1/(d_ij^2)^{power/2}] = -power * (x_i - x_j) / (d_ij^2)^{power/2 + 1}
    # For the full sum, gradient for point i:
    # sum_j -power * (x_i - x_j) / (d_ij^2)^{power/2 + 1}
    
    inv_dist_pow = 1.0 / (dist_sq + eps) ** (power / 2.0 + 1.0)
    np.fill_diagonal(inv_dist_pow, 0.0)
    
    grad = -power * np.sum(diff * inv_dist_pow[:, :, np.newaxis], axis=1)
    
    # Boundary gradient
    boundary_dist_x = 0.5 - np.abs(positions[:, 0])
    boundary_dist_y = 0.5 - np.abs(positions[:, 1])
    
    # For boundary: d/dx [1/(0.5-|x|)^power] = power * sign(x) / (0.5-|x|)^{power+1}
    # when |x| < 0.5
    bx = positions[:, 0]
    by = positions[:, 1]
    
    dbx = np.zeros(n)
    dby = np.zeros(n)
    
    mask_x = boundary_dist_x > eps
    mask_y = boundary_dist_y > eps
    
    dbx[mask_x] = power * np.sign(bx[mask_x]) / (boundary_dist_x[mask_x] ** (power + 1))
    dby[mask_y] = power * np.sign(by[mask_y]) / (boundary_dist_y[mask_y] ** (power + 1))
    
    grad[:, 0] += 10.0 * dbx
    grad[:, 1] += 10.0 * dby
    
    return grad

def _optimize_inverse_power(positions, power=6, max_iter=200, ftol=1e-9, gtol=1e-6):
    """
    Optimize using L-BFGS-B on inverse-power-sum energy.
    """
    n = len(positions)
    bounds = [(-0.499, 0.499)] * (2 * n)
    
    def _objective_and_grad(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        val = _inverse_power_energy(pos, power=power)
        grad = _inverse_power_grad(pos, power=power)
        return val, grad.ravel()
    
    try:
        res = minimize(
            _objective_and_grad,
            positions.ravel(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 5, 'ftol': ftol, 'gtol': gtol}
        )
        return res.x.reshape(-1, 2)
    except Exception:
        return positions.copy()

def _local_push_refine(positions, n_iters=500, push_factor=0.001):
    """
    Iteratively find the closest pair and push them apart, also handle boundaries.
    """
    positions = positions.copy()
    n = len(positions)
    
    for it in range(n_iters):
        # Find closest pair
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        min_idx = np.argmin(dist_sq)
        i, j = np.unravel_index(min_idx, (n, n))
        if i >= j:
            i, j = j, i
        
        d = np.sqrt(dist_sq[i, j])
        if d < 1e-10:
            positions[i, 0] += 0.001 * np.random.randn()
            positions[i, 1] += 0.001 * np.random.randn()
            positions[j, 0] -= 0.001 * np.random.randn()
            positions[j, 1] -= 0.001 * np.random.randn()
        else:
            direction = (positions[i] - positions[j]) / d
            # Push amount proportional to how close they are
            push = push_factor * (1.0 - d / (2 * 0.0366))
            push = max(push, push_factor * 0.1)
            positions[i] += direction * push
            positions[j] -= direction * push
        
        # Handle boundaries
        for k in range(n):
            x, y = positions[k]
            if x > 0.499:
                positions[k, 0] = 0.499
            elif x < -0.499:
                positions[k, 0] = -0.499
            if y > 0.499:
                positions[k, 1] = 0.499
            elif y < -0.499:
                positions[k, 1] = -0.499
    
    return positions

def _multi_scale_optimize(positions, time_budget=60.0):
    """
    Multi-scale optimization: start with high power (focus on closest pairs),
    then reduce power for global refinement.
    """
    start_time = time.time()
    current = positions.copy()
    
    # Start with high power to focus on the most critical pairs
    powers = [12, 8, 6, 4, 3]
    
    for p in powers:
        if time.time() - start_time > time_budget:
            break
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        max_iter = min(100, int(remaining * 10))
        current = _optimize_inverse_power(current, power=p, max_iter=max_iter, ftol=1e-10, gtol=1e-7)
        # Local push after each scale
        current = _local_push_refine(current, n_iters=100, push_factor=0.0005)
    
    return current

def _random_jitter(positions, sigma=0.005):
    """Add small random jitter to escape local minima."""
    jittered = positions + sigma * np.random.randn(*positions.shape)
    return np.clip(jittered, -0.495, 0.495)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_positions = None
    
    # Phase 1: Generate initial hexagonal grid
    hex_pos = _generate_hexagonal_grid(N)
    curr_score = _score(hex_pos)
    if curr_score > best_score:
        best_score = curr_score
        best_positions = hex_pos.copy()
    
    # Phase 2: Multi-scale inverse-power optimization
    if time.time() < deadline - 40:
        try:
            time_left = min(50, deadline - time.time() - 10)
            opt_pos = _multi_scale_optimize(best_positions, time_budget=time_left)
            opt_score = _score(opt_pos)
            if opt_score > best_score:
                best_score = opt_score
                best_positions = opt_pos.copy()
        except Exception:
            pass
    
    # Phase 3: Local push refinement
    if time.time() < deadline - 20:
        try:
            pushed_pos = _local_push_refine(best_positions, n_iters=300, push_factor=0.0008)
            push_score = _score(pushed_pos)
            if push_score > best_score:
                best_score = push_score
                best_positions = pushed_pos.copy()
        except Exception:
            pass
    
    # Phase 4: Try a few restarts with jitter if time permits
    if time.time() < deadline - 15:
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            try:
                jittered = _random_jitter(best_positions, sigma=0.003)
                time_left = min(15, deadline - time.time() - 5)
                if time_left < 3:
                    break
                opt_pos = _multi_scale_optimize(jittered, time_budget=time_left)
                opt_score = _score(opt_pos)
                if opt_score > best_score:
                    best_score = opt_score
                    best_positions = opt_pos.copy()
            except Exception:
                pass
    
    # Phase 5: Final fine-tuning with low power
    if time.time() < deadline - 5:
        try:
            time_left = min(8, deadline - time.time() - 2)
            if time_left > 2:
                final_pos = _optimize_inverse_power(best_positions, power=3, max_iter=50, ftol=1e-10, gtol=1e-7)
                final_score = _score(final_pos)
                if final_score > best_score:
                    best_score = final_score
                    best_positions = final_pos.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_positions is None:
        best_positions = _generate_hexagonal_grid(N)
        
    best_positions = np.clip(best_positions, -0.5, 0.5)
    
    return best_positions

# EVOLVE-BLOCK-END
