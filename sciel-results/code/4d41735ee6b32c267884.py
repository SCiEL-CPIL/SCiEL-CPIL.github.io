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
    
    # Boundary distance
    min_boundary = np.min(np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    ))
    
    if min_boundary <= 0:
        return 0.0

    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(min_boundary, min_pair / 2.0)

def _generate_hexagonal_grid(n_circles, spacing=None):
    """Generate a hexagonal grid that fits well in the unit square."""
    if spacing is None:
        spacing = 0.0735  # slightly larger than 1/sqrt(200*2/sqrt(3)) to ensure fit
    
    points = []
    d = spacing
    max_extent = 1.0
    n_rows = int(max_extent / (d * np.sqrt(3) / 2)) + 2
    
    for row in range(-n_rows, n_rows + 1):
        offset = (row % 2) * 0.5 * d
        x_max = int(max_extent / d) + 1
        for col in range(-x_max, x_max + 1):
            x = col * d + offset
            y = row * np.sqrt(3) / 2 * d
            # Only keep points inside the square with a small margin
            if abs(x) <= 0.49 and abs(y) <= 0.49:
                points.append([x, y])
            
    points = np.array(points)
    
    if len(points) < n_circles:
        # Need more points, use smaller spacing
        spacing = 0.068
        d = spacing
        points = []
        n_rows = int(max_extent / (d * np.sqrt(3) / 2)) + 2
        for row in range(-n_rows, n_rows + 1):
            offset = (row % 2) * 0.5 * d
            x_max = int(max_extent / d) + 1
            for col in range(-x_max, x_max + 1):
                x = col * d + offset
                y = row * np.sqrt(3) / 2 * d
                if abs(x) <= 0.49 and abs(y) <= 0.49:
                    points.append([x, y])
        points = np.array(points)
    
    # Sort by distance to center and pick the n_circles closest
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)[:n_circles]
    points = points[idx]
    
    return points

def _inverse_power_energy(positions, p=8, eps=1e-8):
    """
    Compute the inverse power sum energy.
    Lower energy means larger minimum distance.
    """
    n = len(positions)
    
    # Boundary term: penalize proximity to boundary
    # Use (0.5 - dist_to_boundary)^-p
    boundary_dists = np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    )
    # Add epsilon to avoid division by zero
    boundary_dists = np.maximum(boundary_dists, eps)
    boundary_energy = np.sum(boundary_dists ** (-p))
    
    # Pairwise term
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
    dist = np.sqrt(dist_sq)
    dist = np.maximum(dist, eps)
    
    # Only sum upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    pair_energy = np.sum(pair_dists ** (-p))
    
    return boundary_energy + pair_energy

def _optimize_inverse_power(positions, p=8, max_iter=200, tol=1e-8):
    """
    Optimize positions to minimize inverse power energy.
    """
    n = len(positions)
    bounds = [(-0.5, 0.5)] * (2 * n)
    
    def _obj(pos_flat):
        pos = pos_flat.reshape(-1, 2)
        # Clip to bounds to ensure validity during optimization
        pos = np.clip(pos, -0.5, 0.5)
        return _inverse_power_energy(pos, p=p)
    
    try:
        res = minimize(
            _obj,
            positions.ravel(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6, 'maxfun': max_iter * 10}
        )
        return res.x.reshape(-1, 2)
    except Exception:
        return positions

def _soft_min_radius(positions, t):
    """
    Compute the soft-minimum of the radius constraints.
    """
    n = len(positions)
    
    # Boundary distances
    boundary_dists = np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    )
    
    # Pairwise distances / 2
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    pair_radii = np.sqrt(dist_sq) / 2.0
    
    # Combine all radii
    iu = np.triu_indices(n, k=1)
    pair_radii_flat = pair_radii[iu]
    
    all_radii = np.concatenate([boundary_dists, pair_radii_flat])
    
    # Soft min using log-sum-exp
    x = -all_radii / t
    max_x = np.max(x)
    if not np.isfinite(max_x):
        return 0.0
        
    log_sum = np.log(np.sum(np.exp(x - max_x))) + max_x
    soft_min_val = -t * log_sum
    
    return soft_min_val

def _optimize_soft_min(positions, t_start=5.0, t_end=0.05, n_steps=8, max_iter=30):
    """
    Optimize using L-BFGS-B on negative soft-min radius with annealing.
    """
    current_pos = positions.copy()
    n = len(current_pos)
    bounds = [(-0.5, 0.5)] * (2 * n)
    
    for i in range(n_steps):
        t_val = t_start * (t_end / t_start) ** (i / (n_steps - 1)) if n_steps > 1 else t_start
        
        def _objective(pos_flat):
            pos = pos_flat.reshape(-1, 2)
            return -_soft_min_radius(pos, t_val)
            
        try:
            res = minimize(
                _objective,
                current_pos.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-7, 'gtol': 1e-4}
            )
            current_pos = res.x.reshape(-1, 2)
        except Exception:
            pass
            
    return current_pos

def _pairwise_push_optimize(positions, max_iters=2000):
    """
    Iteratively find the closest pair and push them apart.
    """
    positions = positions.copy()
    n = len(positions)
    
    for it in range(max_iters):
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
            # Random push
            positions[i, 0] += 0.001 * np.random.randn()
            positions[i, 1] += 0.001 * np.random.randn()
            positions[j, 0] -= 0.001 * np.random.randn()
            positions[j, 1] -= 0.001 * np.random.randn()
        else:
            # Push apart along the line connecting them
            direction = (positions[i] - positions[j]) / d
            push = 0.001
            positions[i] += direction * push
            positions[j] -= direction * push
        
        # Also check boundary
        for k in range(n):
            x, y = positions[k]
            if x > 0.49:
                positions[k, 0] = 0.49
            elif x < -0.49:
                positions[k, 0] = -0.49
            if y > 0.49:
                positions[k, 1] = 0.49
            elif y < -0.49:
                positions[k, 1] = -0.49
    
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_positions = None
    
    # Phase 1: Hexagonal initialization
    hex_pos = _generate_hexagonal_grid(N)
    curr_score = _score(hex_pos)
    if curr_score > best_score:
        best_score = curr_score
        best_positions = hex_pos.copy()
    
    # Phase 2: Inverse power energy optimization (smooth surrogate)
    if time.time() < deadline - 60:
        try:
            # Use a moderate power to balance smoothness and sensitivity
            opt_pos = _optimize_inverse_power(best_positions, p=6, max_iter=100, tol=1e-7)
            opt_score = _score(opt_pos)
            if opt_score > best_score:
                best_score = opt_score
                best_positions = opt_pos.copy()
        except Exception:
            pass
    
    # Phase 3: Soft-min optimization with annealing
    if time.time() < deadline - 40:
        try:
            opt_pos = _optimize_soft_min(best_positions, t_start=3.0, t_end=0.05, n_steps=6, max_iter=30)
            opt_score = _score(opt_pos)
            if opt_score > best_score:
                best_score = opt_score
                best_positions = opt_pos.copy()
        except Exception:
            pass
    
    # Phase 4: Pairwise push refinement
    if time.time() < deadline - 20:
        try:
            pushed_pos = _pairwise_push_optimize(best_positions, max_iters=1500)
            push_score = _score(pushed_pos)
            if push_score > best_score:
                best_score = push_score
                best_positions = pushed_pos.copy()
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish with inverse power (higher p)
    if time.time() < deadline - 10:
        try:
            polish_pos = _optimize_inverse_power(best_positions, p=12, max_iter=50, tol=1e-9)
            polish_score = _score(polish_pos)
            if polish_score > best_score:
                best_score = polish_score
                best_positions = polish_pos.copy()
        except Exception:
            pass
    
    # Phase 6: Final soft-min polish with very low temperature
    if time.time() < deadline - 5:
        try:
            polish_pos = _optimize_soft_min(best_positions, t_start=0.1, t_end=0.01, n_steps=3, max_iter=20)
            polish_score = _score(polish_pos)
            if polish_score > best_score:
                best_score = polish_score
                best_positions = polish_pos.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_positions is None:
        best_positions = _generate_hexagonal_grid(N)
        
    best_positions = np.clip(best_positions, -0.5, 0.5)
    
    return best_positions

# EVOLVE-BLOCK-END
