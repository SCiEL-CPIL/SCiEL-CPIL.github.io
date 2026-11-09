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

def _generate_hexagonal_grid(n_circles):
    """Generate a hexagonal grid that fits well in the unit square."""
    points = []
    # Target radius ~0.036, so spacing ~0.072
    # Use a spacing that gives roughly the right density
    d = 0.072
    
    # Generate points in a larger hexagonal region
    max_extent = 1.0
    n_rows = int(max_extent / (d * np.sqrt(3) / 2)) + 2
    
    for row in range(-n_rows, n_rows + 1):
        offset = (row % 2) * 0.5 * d
        x_max = int(max_extent / d) + 1
        for col in range(-x_max, x_max + 1):
            x = col * d + offset
            y = row * np.sqrt(3) / 2 * d
            # Only keep points inside the square with margin
            if abs(x) <= 0.48 and abs(y) <= 0.48:
                points.append([x, y])
            
    points = np.array(points)
    
    if len(points) < n_circles:
        # Need more points, use smaller spacing
        d = 0.065
        points = []
        n_rows = int(max_extent / (d * np.sqrt(3) / 2)) + 2
        for row in range(-n_rows, n_rows + 1):
            offset = (row % 2) * 0.5 * d
            x_max = int(max_extent / d) + 1
            for col in range(-x_max, x_max + 1):
                x = col * d + offset
                y = row * np.sqrt(3) / 2 * d
                if abs(x) <= 0.48 and abs(y) <= 0.48:
                    points.append([x, y])
        points = np.array(points)
    
    # Sort by distance to center and pick the n_circles closest
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)[:n_circles]
    points = points[idx]
    
    return points

def _force_relax(positions, max_iters=2000, dt=0.001, k_repel=1.0, k_boundary=10.0):
    """
    Force-based relaxation using repulsive forces between circles
    and attractive forces toward the boundary.
    """
    positions = positions.copy()
    n = len(positions)
    
    for it in range(max_iters):
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Pairwise repulsive forces
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
        dist = np.sqrt(dist_sq)
        
        # Force magnitude: k / dist^2 (Coulomb-like)
        # Only apply for pairs closer than some threshold to save time
        threshold = 0.15
        mask = dist < threshold
        mask &= ~np.eye(n, dtype=bool)
        
        # Compute forces from close pairs
        for i in range(n):
            close_j = np.where(mask[i])[0]
            for j in close_j:
                d = dist[i, j]
                if d < 1e-10:
                    continue
                # Repulsive force: pushes i away from j
                force_mag = k_repel / (d * d)
                direction = diff[i, j] / d
                forces[i] += force_mag * direction
        
        # Boundary forces: push points away from boundaries
        for i in range(n):
            x, y = positions[i]
            # Distance to each boundary
            d_left = x + 0.5
            d_right = 0.5 - x
            d_bottom = y + 0.5
            d_top = 0.5 - y
            
            # Force proportional to 1/d^2 when close to boundary
            if d_left < 0.1:
                forces[i, 0] += k_boundary / (d_left * d_left)
            if d_right < 0.1:
                forces[i, 0] -= k_boundary / (d_right * d_right)
            if d_bottom < 0.1:
                forces[i, 1] += k_boundary / (d_bottom * d_bottom)
            if d_top < 0.1:
                forces[i, 1] -= k_boundary / (d_top * d_top)
        
        # Update positions with damping
        positions += dt * forces
        positions = np.clip(positions, -0.49, 0.49)
        
        # Check convergence
        if it > 100:
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force < 1e-6:
                break
    
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

def _optimize_soft_min(positions, bounds, t_start=10.0, t_end=0.1, n_steps=10, max_iter=50):
    """
    Optimize using L-BFGS-B on negative soft-min radius with annealing.
    """
    current_pos = positions.copy()
    
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

def _pairwise_push_optimize(positions, max_iters=5000):
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
            push = 0.002
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
    
    # Phase 2: Force-based relaxation
    if time.time() < deadline - 50:
        try:
            relaxed_pos = _force_relax(best_positions, max_iters=1500, dt=0.002, k_repel=0.5, k_boundary=5.0)
            relax_score = _score(relaxed_pos)
            if relax_score > best_score:
                best_score = relax_score
                best_positions = relaxed_pos.copy()
        except Exception:
            pass
    
    # Phase 3: Soft-min optimization
    if time.time() < deadline - 30:
        bounds = [(-0.5, 0.5)] * (2 * N)
        try:
            opt_pos = _optimize_soft_min(best_positions, bounds, t_start=5.0, t_end=0.05, n_steps=8, max_iter=30)
            opt_score = _score(opt_pos)
            if opt_score > best_score:
                best_score = opt_score
                best_positions = opt_pos.copy()
        except Exception:
            pass
    
    # Phase 4: Pairwise push refinement
    if time.time() < deadline - 15:
        try:
            pushed_pos = _pairwise_push_optimize(best_positions, max_iters=3000)
            push_score = _score(pushed_pos)
            if push_score > best_score:
                best_score = push_score
                best_positions = pushed_pos.copy()
        except Exception:
            pass
    
    # Phase 5: Final soft-min polish
    if time.time() < deadline - 5:
        bounds = [(-0.5, 0.5)] * (2 * N)
        try:
            polish_pos = _optimize_soft_min(best_positions, bounds, t_start=0.1, t_end=0.01, n_steps=5, max_iter=20)
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
