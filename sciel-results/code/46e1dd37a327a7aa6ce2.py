"""Structural change: Direct bottleneck-elimination + simulated annealing local search."""
N = 200
import numpy as np
import time
import random

_DEADLINE = 145.0
_START_TIME = time.time()

def _check_time():
    return time.time() - _START_TIME < _DEADLINE

def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)

def _get_radius_and_bottleneck(centers):
    """
    Returns (r, bottleneck_info) where bottleneck_info is a dict describing
    the constraint that limits the radius.
    If r is limited by boundary: {'type': 'boundary', 'idx': i, 'axis': 0/1}
    If r is limited by pair: {'type': 'pair', 'i': i, 'j': j}
    """
    boundary = np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    )
    min_bound_val = np.min(boundary)
    min_bound_idx = np.argmin(boundary)
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_val = np.min(dists) / 2.0
    
    r = min(min_bound_val, min_pair_val)
    
    if min_bound_val <= min_pair_val:
        # Boundary is the bottleneck
        # Determine which axis
        x_margin = 0.5 - np.abs(centers[min_bound_idx, 0])
        y_margin = 0.5 - np.abs(centers[min_bound_idx, 1])
        axis = 0 if x_margin <= y_margin else 1
        return r, {'type': 'boundary', 'idx': min_bound_idx, 'axis': axis}
    else:
        # Pair is the bottleneck
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        if i > j:
            i, j = j, i
        return r, {'type': 'pair', 'i': i, 'j': j}

def _generate_hex_initial(n_points):
    """Generate a hexagonal grid of points."""
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    points = []
    row_height = s * np.sqrt(3) / 2
    n_rows = int(1.0 / row_height) + 2
    n_cols = int(1.0 / s) + 2
    
    total_width = (n_cols - 1) * s
    total_height = (n_rows - 1) * row_height
    
    start_x = -0.5 + (1.0 - total_width) / 2
    start_y = -0.5 + (1.0 - total_height) / 2
    
    for i in range(n_rows):
        for j in range(n_cols):
            x = start_x + j * s
            if i % 2 == 1:
                x += s / 2
            y = start_y + i * row_height
            
            if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) > n_points:
        center_dist = np.sum(points ** 2, axis=1)
        idx = np.argsort(center_dist)[:n_points]
        points = points[idx]
        
    if len(points) < n_points:
        extra = np.random.uniform(-0.45, 0.45, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])
        
    return points[:n_points]

def _local_search(centers, max_iters=5000):
    """
    Direct bottleneck elimination.
    Find the bottleneck constraint, move the involved points to relieve it.
    """
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    for _ in range(max_iters):
        if not _check_time():
            break
            
        r, info = _get_radius_and_bottleneck(best_centers)
        
        # Try to move points to improve the bottleneck
        # We define a move that increases the margin of the bottleneck constraint
        # while trying not to decrease other margins too much.
        
        if info['type'] == 'pair':
            i, j = info['i'], info['j']
            c_i = best_centers[i]
            c_j = best_centers[j]
            vec = c_i - c_j
            dist = np.linalg.norm(vec)
            if dist < 1e-9:
                continue
            unit_vec = vec / dist
            
            # Move i in direction of unit_vec, j in direction of -unit_vec
            # Step size: try to increase distance by delta
            # We want new_dist > old_dist.
            # Let's try a few step sizes.
            step_sizes = [0.001, 0.0005, 0.002, 0.0001]
            improved = False
            for step in step_sizes:
                new_centers = best_centers.copy()
                new_centers[i] += unit_vec * step
                new_centers[j] -= unit_vec * step
                
                # Clip
                new_centers[:, 0] = np.clip(new_centers[:, 0], -0.49, 0.49)
                new_centers[:, 1] = np.clip(new_centers[:, 1], -0.49, 0.49)
                
                new_r = _compute_radius(new_centers)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_centers = new_centers
                    improved = True
                    break
            
            if not improved:
                # If simple move didn't work, try moving one point in a random direction
                # that increases distance to the other, but allows rotation.
                # Actually, let's just break if no improvement found for this bottleneck type
                # to avoid getting stuck in a local minimum where moving along the line is bad.
                # We can try a perpendicular move? No, that decreases distance.
                # We need to move them apart. If moving apart along the line fails,
                # it's likely because it hits a wall or another circle.
                # In that case, we need a more complex move (e.g. move i, then j, or move both in a rotated frame).
                # For simplicity, we'll just break the local search loop and rely on SA for global moves.
                # Or, we can try a small random perturbation of the pair.
                pass
                
        elif info['type'] == 'boundary':
            idx = info['idx']
            axis = info['axis']
            c = best_centers[idx]
            
            # Move towards center
            # Direction is -sign(c[axis]) * e_axis
            direction = np.zeros(2)
            if c[axis] > 0:
                direction[axis] = -1.0
            elif c[axis] < 0:
                direction[axis] = 1.0
            else:
                # On center line, move randomly?
                direction[axis] = 1.0 if random.random() > 0.5 else -1.0
                
            step_sizes = [0.001, 0.0005, 0.002, 0.0001]
            improved = False
            for step in step_sizes:
                new_centers = best_centers.copy()
                new_centers[idx] += direction * step
                
                new_centers[:, 0] = np.clip(new_centers[:, 0], -0.49, 0.49)
                new_centers[:, 1] = np.clip(new_centers[:, 1], -0.49, 0.49)
                
                new_r = _compute_radius(new_centers)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_centers = new_centers
                    improved = True
                    break
            
            if not improved:
                pass
                
    return best_centers, best_r

def _sa_local_search(centers, max_iters=20000):
    """
    Simulated Annealing on the exact radius.
    Moves: pick a random point, move it by a small random vector.
    Accept if r increases, or with probability exp((r_new - r_old) / T).
    """
    current_centers = centers.copy()
    current_r = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r
    
    T = 0.001
    T_min = 1e-6
    cooling = 0.999
    
    for _ in range(max_iters):
        if not _check_time():
            break
            
        # Pick a random point
        idx = random.randint(0, N - 1)
        
        # Generate a random move
        # Scale of move should be related to current radius or just fixed small
        # Let's use a fixed small scale
        delta = np.random.normal(0, 0.005, size=2)
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        new_centers[idx, 0] = np.clip(new_centers[idx, 0], -0.49, 0.49)
        new_centers[idx, 1] = np.clip(new_centers[idx, 1], -0.49, 0.49)
        
        new_r = _compute_radius(new_centers)
        
        delta_r = new_r - current_r
        
        if delta_r > 0 or random.random() < np.exp(delta_r / T):
            current_centers = new_centers
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
                
        T *= cooling
        
    return best_centers, best_r

def construct():
    start_time = time.time()
    
    # 1. Initialize
    centers = _generate_hex_initial(N)
    centers += np.random.normal(0, 0.001, size=centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], -0.49, 0.49)
    centers[:, 1] = np.clip(centers[:, 1], -0.49, 0.49)
    
    # 2. Local Search (Bottleneck Elimination)
    centers, r_best = _local_search(centers, max_iters=10000)
    
    # 3. Simulated Annealing
    if _check_time():
        centers, r_sa = _sa_local_search(centers, max_iters=30000)
        if r_sa > r_best:
            r_best = r_sa
            # Re-run local search from the SA result?
            # SA might have moved things slightly, local search can polish.
            centers, r_polish = _local_search(centers, max_iters=5000)
            if r_polish > r_best:
                r_best = r_polish
                
    # 4. Final validation
    r_final = _compute_radius(centers)
    if r_final <= 0:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N]
        
    return centers

