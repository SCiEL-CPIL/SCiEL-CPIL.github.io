"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _radius_for_centers(centers):
    """Compute the radius: min of half pairwise distances and wall distances."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances: min over each coordinate of (0.5 - |c|)
    wall = 0.5 - np.abs(centers)
    min_wall = np.min(wall)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_wall, min_pair)


def _multi_push_apart(centers, iterations=200, step_scale=1.0, threshold_factor=1.5):
    """Iteratively push apart ALL pairs that are too close."""
    centers = centers.copy()
    for it in range(iterations):
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Current minimum distance
        min_dist = np.min(dist)
        
        # Threshold: push apart pairs closer than threshold_factor * min_dist
        threshold = threshold_factor * min_dist
        
        # Find all pairs below threshold
        mask = (dist < threshold) & (dist > 1e-12)
        np.fill_diagonal(mask, False)
        
        # Also check wall constraints
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        eff_radius = min(min_dist / 2.0, min_wall)
        
        # Determine if pair or wall is bottleneck
        pair_bottleneck = (min_dist / 2.0) <= (min_wall + 1e-12)
        
        # Build force vectors
        forces = np.zeros_like(centers)
        
        if pair_bottleneck:
            # Push apart all close pairs
            for i in range(N):
                for j in range(i + 1, N):
                    if mask[i, j]:
                        d = dist[i, j]
                        if d < 1e-12:
                            continue
                        direction = (centers[i] - centers[j]) / d
                        # Force proportional to how much below threshold
                        strength = (threshold - d) / threshold
                        force_mag = 0.005 * step_scale * strength
                        forces[i] += direction * force_mag
                        forces[j] -= direction * force_mag
        else:
            # Wall is bottleneck: push spheres away from nearest walls
            for i in range(N):
                for coord in range(3):
                    wall_dist = wall[i, coord]
                    if wall_dist < threshold / 2.0:
                        # Push away from this wall
                        if centers[i, coord] > 0:
                            forces[i, coord] -= 0.005 * step_scale
                        else:
                            forces[i, coord] += 0.005 * step_scale
        
        # Apply forces with adaptive step
        step = 0.1 * step_scale
        centers += forces * step
        
        # Clip to stay in cube
        centers = np.clip(centers, -0.499, 0.499)
        
        # Early termination if no improvement
        new_min_dist = np.min(dist)  # approximate
        if new_min_dist <= min_dist + 1e-10:
            if it > 10:
                break
    
    return centers


def _epigraph_objective(x):
    """
    Epigraph formulation: minimize -t subject to all constraints >= t.
    x = [centers (N*3), t]
    We use a smooth penalty for constraint violations.
    """
    centers = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    # Wall constraints: 0.5 - |c_ik| >= t
    wall = 0.5 - np.abs(centers)
    wall_violations = np.maximum(0, t - wall)
    
    # Pair constraints: |c_i - c_j|/2 >= t
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_violations = np.maximum(0, t - dist / 2.0)
    
    # Smooth penalty: sum of squared violations
    penalty = np.sum(wall_violations ** 2) + np.sum(pair_violations ** 2)
    
    # Objective: -t + penalty
    return -t + penalty


def _epigraph_gradient(x):
    """Gradient of the epigraph objective (numerical, for L-BFGS-B)."""
    # L-BFGS-B will use numerical gradient if we don't provide one
    # But we can provide an analytical one for speed
    centers = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    grad = np.zeros_like(x)
    
    # Wall part
    wall = 0.5 - np.abs(centers)
    for i in range(N):
        for coord in range(3):
            if t > wall[i, coord]:
                # d/dx of (t - wall)^2 = -2*(t-wall)*d(wall)/dx
                # wall = 0.5 - |c|, so d(wall)/d(c) = -sign(c)
                sign = np.sign(centers[i, coord])
                if sign == 0:
                    sign = 1.0
                grad[i*3 + coord] += -2 * (t - wall[i, coord]) * (-sign)
    
    # Pair part
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = dist[i, j]
            if t > d / 2.0 and d > 1e-12:
                violation = t - d / 2.0
                # d/d(c_i) of (t - d/2)^2 = -2*(t-d/2) * d(d/2)/d(c_i)
                # d(d)/d(c_i) = (c_i - c_j) / d
                direction = (centers[i] - centers[j]) / d
                grad[i*3:i*3+3] += -2 * violation * (direction / 2.0)
                grad[j*3:j*3+3] += -2 * violation * (-direction / 2.0)
    
    # Gradient with respect to t: -1 + 2*sum(violations)
    wall_violations = np.maximum(0, t - (0.5 - np.abs(centers)))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_violations = np.maximum(0, t - dist / 2.0)
    grad[N*3] = -1.0 + 2.0 * (np.sum(wall_violations) + np.sum(pair_violations))
    
    return grad


def _make_structured_initial(seed):
    """
    Create a structured initialization based on known good packings.
    50 spheres in unit cube: try a 5x5x4 grid with some positions removed/adjusted.
    """
    rng = np.random.RandomState(seed)
    
    # Base: 5x5x4 = 100 positions, we need 50
    # Use a pattern that removes every other sphere in a checkerboard-like fashion
    m1, m2, m3 = 5, 5, 4
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    
    # Generate all 100 positions
    all_positions = []
    for z in ticks3:
        for y in ticks2:
            for x in ticks1:
                all_positions.append([x, y, z])
    all_positions = np.array(all_positions)
    
    # Select 50 positions: use a pattern
    # Option 1: every other in a 3D checkerboard
    indices = []
    for idx, pos in enumerate(all_positions):
        # Compute a hash-like index
        i = (pos[0] + 0.5) * m1
        j = (pos[1] + 0.5) * m2
        k = (pos[2] + 0.5) * m3
        if (int(round(i)) + int(round(j)) + int(round(k))) % 2 == 0:
            indices.append(idx)
    
    # Take first 50 from the selected
    if len(indices) >= N:
        centers = all_positions[indices[:N]].copy()
    else:
        # Fallback: take first 50
        centers = all_positions[:N].copy()
    
    # Add small jitter
    centers += rng.uniform(-0.003, 0.003, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _make_grid_initial(seed):
    """Create a jittered grid initialization."""
    rng = np.random.RandomState(seed)
    # Use a 4x4x4 grid = 64 positions, take first 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small random jitter
    centers += rng.uniform(-0.005, 0.005, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _make_random_initial(seed):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.49, 0.49, size=(N, 3))


def _optimize_local(centers, time_limit=10.0):
    """Optimize a configuration using multi-push-apart and epigraph L-BFGS-B."""
    start = time.time()
    current = centers.copy()
    current_radius = _radius_for_centers(current)
    
    # Phase 1: Multi-push-apart refinement
    for scale in [1.0, 0.5, 0.25, 0.1, 0.05]:
        if time.time() - start > time_limit * 0.35:
            break
        new_centers = _multi_push_apart(current, iterations=150, step_scale=scale, threshold_factor=1.3)
        new_radius = _radius_for_centers(new_centers)
        if new_radius > current_radius + 1e-12:
            current = new_centers
            current_radius = new_radius
    
    # Phase 2: Epigraph optimization with L-BFGS-B
    if time.time() - start < time_limit * 0.75:
        # Initialize t with current radius
        t0 = current_radius
        x0 = np.concatenate([current.flatten(), [t0]])
        
        # Bounds for centers and t
        bounds = [(-0.499, 0.499)] * (N * 3) + [(0.0, 0.499)]
        
        try:
            result = minimize(
                _epigraph_objective,
                x0,
                jac=_epigraph_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
            )
            new_centers = result.x[:N*3].reshape(N, 3)
            new_t = result.x[N*3]
            new_radius = _radius_for_centers(new_centers)
            if new_radius > current_radius + 1e-12:
                current = new_centers
                current_radius = new_radius
        except Exception:
            pass
    
    # Phase 3: Final polish with high-precision push-apart
    if time.time() - start < time_limit * 0.95:
        for scale in [0.02, 0.01, 0.005]:
            if time.time() - start > time_limit * 0.98:
                break
            new_centers = _multi_push_apart(current, iterations=50, step_scale=scale, threshold_factor=1.1)
            new_radius = _radius_for_centers(new_centers)
            if new_radius > current_radius + 1e-14:
                current = new_centers
                current_radius = new_radius
    
    return current, current_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_centers = None
    best_radius = 0.0

    # Generate diverse initializations
    initializations = []
    
    # Structured initializations (checkerboard pattern from 5x5x4)
    for seed in [42, 123, 456, 789]:
        initializations.append(_make_structured_initial(seed))
    
    # Grid-based initializations with different jitters
    for seed in [1, 2, 3]:
        initializations.append(_make_grid_initial(seed))
    
    # Random initializations
    for seed in [10, 20]:
        initializations.append(_make_random_initial(seed))

    # Optimize each initialization
    for init_idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 15.0:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        n_remaining = len(initializations) - init_idx
        time_per_init = min(12.0, remaining / (n_remaining + 1))
        
        if time_per_init < 3.0:
            break
        
        try:
            optimized, radius = _optimize_local(init_centers, time_limit=time_per_init)
        except Exception:
            continue
        
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized.copy()
    
    # Final refinement: take the best solution and do additional polish
    if best_centers is not None and time.time() < deadline - 5.0:
        time_left = deadline - time.time() - 5.0
        if time_left > 3.0:
            try:
                final_centers, final_radius = _optimize_local(best_centers, time_limit=time_left)
                if final_radius > best_radius:
                    best_radius = final_radius
                    best_centers = final_centers
            except Exception:
                pass

    if best_centers is None:
        # Fallback: simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.499, 0.499)
    return best_centers
# EVOLVE-BLOCK-END
