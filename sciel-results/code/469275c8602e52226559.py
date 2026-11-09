"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 140.0  # Safety margin before 150s limit
_start_time = time.time()


def _time_left():
    return DEADLINE - (time.time() - _start_time)


def _compute_radius(centers):
    """Compute the score: largest common radius for given centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-10):
        return 0.0
    # Vectorized pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _forces_and_energy(centers, r_target):
    """
    Physics-based force computation.
    Returns forces on each particle and a potential energy.
    
    Forces:
    - Repulsive force between pairs: if dist < 2*r_target, push apart
    - Wall constraint: if center too close to wall, push inward
    """
    forces = np.zeros_like(centers)
    
    # Pairwise repulsive forces
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)  # Avoid division by zero
    
    # Repulsive force: F = k * (1/dist^2) * direction for dist < 2*r
    # Use a smooth potential: U = k * (2*r - dist)^2 for dist < 2*r
    # dU/dx = -2*k*(2*r - dist) * d(dist)/dx = -2*k*(2*r - dist) * (x_i - x_j)/dist
    
    # Only compute for pairs with dist < 2*r_target
    mask = (dists < 2.0 * r_target) & (dists > 1e-12)
    if np.any(mask):
        # Force magnitude: proportional to (2*r - dist) / dist
        # This gives a smooth repulsive force
        overlap = np.maximum(2.0 * r_target - dists, 0.0)
        force_mag = overlap / dists
        # Apply force: particle i is pushed away from j
        # F_i += force_mag * (c_i - c_j) / dist = force_mag * diff[i,j] / dists[i,j]
        # But diff[i,j] = c_i - c_j, so direction is already correct
        # Force on i from j: force_mag[i,j] * diff[i,j] / dists[i,j]
        # Sum over j for each i
        f_mag = force_mag / dists  # shape (N, N)
        forces += np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
    
    # Wall forces: push centers away from walls if too close
    # Distance to nearest wall for each center
    for axis in range(2):
        # Left wall: x < -0.5 + r
        left_violation = np.maximum((-0.5 + r_target - centers[:, axis]), 0.0)
        forces[:, axis] += 10.0 * left_violation
        # Right wall: x > 0.5 - r
        right_violation = np.maximum((centers[:, axis] - (0.5 - r_target)), 0.0)
        forces[:, axis] -= 10.0 * right_violation
    
    return forces


def _anneal_optimize(centers, max_iter=5000, tol=1e-9):
    """
    Physics-based annealing optimization.
    Start with a small radius and gradually increase while maintaining non-overlap.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    
    if current_r <= 0:
        current_r = 0.01
    
    # Annealing schedule: start with larger effective radius, decrease
    # We simulate at a target radius slightly above current, with forces
    # pushing circles apart and to walls
    
    best_r = current_r
    best_centers = centers.copy()
    
    # Temperature for random perturbations
    temp = 0.01
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Compute forces at current target radius
        r_target = current_r * 1.01  # Slightly larger to push for improvement
        
        forces = _forces_and_energy(centers, r_target)
        
        # Add small random perturbation (simulated annealing)
        noise = np.random.normal(0, temp, size=centers.shape)
        
        # Update positions
        step = 0.005
        centers = centers + step * forces + noise
        
        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Evaluate
        new_r = _compute_radius(centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            # Decrease temperature on improvement
            temp *= 0.995
        else:
            # Accept with some probability (annealing)
            if np.random.random() < np.exp(-(best_r - new_r) / max(temp, 1e-10)):
                pass  # Accept anyway
            else:
                centers = best_centers.copy()
        
        # Decay temperature
        temp *= 0.999
        if temp < 1e-6:
            temp = 1e-6
    
    return best_centers, best_r


def _gradient_ascent(centers, max_iter=3000, tol=1e-10):
    """
    Gradient ascent on the smooth soft-min objective.
    Uses analytic gradient of the soft-min of pair distances/2 and wall distances.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    
    # Soft-min parameter
    beta = 50.0
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        # Wall distances
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        
        # Soft-min of all constraints:
        # r = min(min_pair/2, min_wall)
        # We use a soft-min over all pair distances/2 and all wall distances
        # Values: pair_dists/2 (N*(N-1)/2 values) and wall_dists (N values)
        
        # For efficiency, use the minimum pair dist and minimum wall dist
        # with a soft combination
        min_pair = np.min(dists)
        min_wall = np.min(wall_dists)
        
        # Gradient of soft-min(a, b) w.r.t. a and b:
        # softmin(a,b) = -log(exp(-a/beta) + exp(-b/beta)) * beta
        # d(softmin)/da = exp(-a/beta) / (exp(-a/beta) + exp(-b/beta))
        # d(softmin)/db = exp(-b/beta) / (exp(-a/beta) + exp(-b/beta))
        
        ea = np.exp(-min_pair / (2.0 * beta))
        eb = np.exp(-min_wall / beta)
        denom = ea + eb
        
        w_pair = ea / denom  # weight for pair term
        w_wall = eb / denom  # weight for wall term
        
        # Gradient w.r.t. centers for pair term:
        # d(min_pair)/d(centers[i]) = sum over j where dist(i,j) is min of (centers[i] - centers[j]) / dist(i,j)
        # Actually, d(dist(i,j))/d(centers[i]) = (centers[i] - centers[j]) / dist(i,j)
        
        grad = np.zeros_like(centers)
        
        # Pair contribution: only the closest pair matters for the gradient
        # Find the closest pair
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        min_idx = np.argmin(pair_dists)
        i, j = iu[0][min_idx], iu[1][min_idx]
        
        d_ij = centers[i] - centers[j]
        dist_ij = dists[i, j]
        if dist_ij > 1e-12:
            grad[i] += w_pair * d_ij / (2.0 * dist_ij)  # d(min_pair/2)/d(centers[i])
            grad[j] -= w_pair * d_ij / (2.0 * dist_ij)  # d(min_pair/2)/d(centers[j])
        
        # Wall contribution: only the closest wall matters
        min_wall_idx = np.argmin(wall_dists)
        ci = centers[min_wall_idx]
        # Determine which wall is closest
        for axis in range(2):
            if ci[axis] >= 0:
                # Right/top wall
                grad[min_wall_idx, axis] -= w_wall
            else:
                # Left/bottom wall
                grad[min_wall_idx, axis] += w_wall
        
        # Gradient ascent: move in direction of increasing r
        step_size = 0.001
        new_centers = centers + step_size * grad
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_r = _compute_radius(new_centers)
        
        if new_r > current_r + 1e-12:
            centers = new_centers
            current_r = new_r
        else:
            step_size *= 0.5
            if step_size < 1e-12:
                break
    
    return centers, current_r


def _hex_initial():
    """Hexagonal packing initialization."""
    centers = np.zeros((N, 2))
    rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    spacing = 1.0 / (rows * 0.866)
    
    idx = 0
    for row in range(rows):
        if idx >= N:
            break
        y = -0.5 + spacing * (row + 0.5) * 0.866
        if abs(y) > 0.5:
            continue
        offset = (row % 2) * spacing * 0.5
        max_cols = int((1.0 - 2 * spacing * 0.25) / spacing) + 1
        for col in range(max_cols):
            if idx >= N:
                break
            x = -0.5 + spacing * (col + 0.5) + offset
            if abs(x) <= 0.5:
                centers[idx] = [x, y]
                idx += 1
        if idx >= N:
            break
    
    if idx < N:
        for i in range(idx, N):
            centers[i] = [np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)]
    
    return centers[:N]


def _grid_initial(m):
    """Grid initialization with m columns."""
    centers = []
    n_rows = int(np.ceil(N / m))
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(n_rows) + 0.5) / n_rows - 0.5
    for y in ys:
        for x in xs:
            centers.append([x, y])
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    while len(centers) < N:
        centers.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
    return np.array(centers[:N])


def _random_initial():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 2))


def construct():
    """
    Construct the centers of 200 equal circles inside [-1/2, 1/2]^2
    to maximize the common radius.
    """
    global _start_time
    _start_time = time.time()
    
    best_centers = None
    best_r = -1.0
    
    # Try multiple initializations
    initializations = []
    
    for m in [10, 11, 12, 13, 14]:
        initializations.append(_grid_initial(m))
    
    initializations.append(_hex_initial())
    
    for _ in range(3):
        initializations.append(_random_initial())
    
    # Process each initialization with both methods
    for init_idx, init_centers in enumerate(initializations):
        if _time_left() < 10.0:
            break
        
        r = _compute_radius(init_centers)
        if r > best_r:
            best_r = r
            best_centers = init_centers.copy()
        
        # Gradient ascent first (faster convergence)
        try:
            opt_centers, opt_r = _gradient_ascent(init_centers, max_iter=2000)
            if opt_r > best_r:
                best_r = opt_r
                best_centers = opt_centers.copy()
        except Exception:
            pass
        
        # Then annealing for further improvement
        if _time_left() > 15.0 and best_centers is not None:
            try:
                anneal_centers, anneal_r = _anneal_optimize(best_centers, max_iter=3000)
                if anneal_r > best_r:
                    best_r = anneal_r
                    best_centers = anneal_centers.copy()
            except Exception:
                pass
    
    # Final polish with gradient ascent on the best
    if best_centers is not None and _time_left() > 10.0:
        try:
            final_centers, final_r = _gradient_ascent(best_centers, max_iter=5000)
            if final_r > best_r:
                best_r = final_r
                best_centers = final_centers.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
