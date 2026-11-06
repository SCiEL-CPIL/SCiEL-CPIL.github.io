"""Rewritten program: bottleneck-driven local optimization for 200 circles in unit square."""
import numpy as np
import time

N = 200
HALF = 0.5

def _compute_radius(centers):
    """Compute the valid radius: min of half pairwise distances and wall distances."""
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(HALF - centers[:, 0], HALF - centers[:, 1]),
        np.minimum(centers[:, 0] + HALF, centers[:, 1] + HALF)
    )
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # For N=200, we can do full pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)


def _compute_radius_fast(centers):
    """Faster radius computation using vectorized ops, no full NxN matrix stored."""
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(HALF - centers[:, 0], HALF - centers[:, 1]),
        np.minimum(centers[:, 0] + HALF, centers[:, 1] + HALF)
    )
    min_wall = np.min(wall_dists)
    
    # Pairwise: compute in chunks to save memory
    n = len(centers)
    min_pair = np.inf
    chunk = 50
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        diff = centers[i:end, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Zero out diagonal for rows in this chunk
        for j in range(end - i):
            dists[j, i + j] = np.inf
        chunk_min = np.min(dists)
        if chunk_min < min_pair:
            min_pair = chunk_min
    min_pair /= 2.0
    
    return min(min_wall, min_pair)


def _find_bottleneck(centers):
    """Find the pair or wall that constrains the radius."""
    wall_dists = np.minimum(
        np.minimum(HALF - centers[:, 0], HALF - centers[:, 1]),
        np.minimum(centers[:, 0] + HALF, centers[:, 1] + HALF)
    )
    min_wall = np.min(wall_dists)
    wall_idx = np.argmin(wall_dists)
    
    n = len(centers)
    min_pair = np.inf
    pair_i, pair_j = 0, 1
    for i in range(n):
        diff = centers[i] - centers
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        dists[i] = np.inf
        idx = np.argmin(dists)
        if dists[idx] < min_pair:
            min_pair = dists[idx]
            pair_i, pair_j = i, idx
    
    min_pair_half = min_pair / 2.0
    
    if min_wall < min_pair_half:
        # Wall is bottleneck
        return ('wall', wall_idx, min_wall)
    else:
        return ('pair', pair_i, pair_j, min_pair_half)


def _grid_init(n):
    """Create initial grid placement."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n].astype(np.float64)


def _optimize(centers, time_limit, deadline):
    """Bottleneck-driven local optimization."""
    best_radius = _compute_radius_fast(centers)
    best_centers = centers.copy()
    
    # Adaptive step size
    step = 0.01
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            step *= 0.7
            if step < 1e-8:
                break
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.3)
        
        # Find bottleneck
        result = _find_bottleneck(centers)
        
        if result[0] == 'wall':
            _, idx, _ = result
            c = centers[idx]
            # Move away from the closest wall
            # Determine which wall
            d_left = c[0] + HALF
            d_right = HALF - c[0]
            d_bottom = c[1] + HALF
            d_top = HALF - c[1]
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            if min_d == d_left:
                dx, dy = step, 0
            elif min_d == d_right:
                dx, dy = -step, 0
            elif min_d == d_bottom:
                dx, dy = 0, step
            else:
                dx, dy = 0, -step
            
            # Try moving in that direction
            new_c = c + np.array([dx, dy])
            # Keep inside
            new_c[0] = np.clip(new_c[0], -HALF, HALF)
            new_c[1] = np.clip(new_c[1], -HALF, HALF)
            
            centers[idx] = new_c
            r = _compute_radius_fast(centers)
            
            if r > best_radius + 1e-12:
                best_radius = r
                best_centers = centers.copy()
                step = min(step * 1.1, 0.05)
                no_improve_count = 0
            else:
                # Revert
                centers[idx] = c
                no_improve_count += 1
        else:
            _, i, j, _ = result
            ci = centers[i].copy()
            cj = centers[j].copy()
            
            # Move i and j apart along their connecting vector
            diff = ci - cj
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                # Random small perturbation
                centers[i] += np.random.randn(2) * step * 0.1
                centers[j] -= np.random.randn(2) * step * 0.1
                r = _compute_radius_fast(centers)
                if r > best_radius + 1e-12:
                    best_radius = r
                    best_centers = centers.copy()
                    step = min(step * 1.1, 0.05)
                    no_improve_count = 0
                else:
                    centers[i] = ci
                    centers[j] = cj
                    no_improve_count += 1
                continue
            
            direction = diff / dist
            # Move i in +direction, j in -direction
            centers[i] = ci + direction * step * 0.5
            centers[j] = cj - direction * step * 0.5
            
            # Keep inside
            centers[i][0] = np.clip(centers[i][0], -HALF, HALF)
            centers[i][1] = np.clip(centers[i][1], -HALF, HALF)
            centers[j][0] = np.clip(centers[j][0], -HALF, HALF)
            centers[j][1] = np.clip(centers[j][1], -HALF, HALF)
            
            r = _compute_radius_fast(centers)
            
            if r > best_radius + 1e-12:
                best_radius = r
                best_centers = centers.copy()
                step = min(step * 1.1, 0.05)
                no_improve_count = 0
            else:
                centers[i] = ci
                centers[j] = cj
                no_improve_count += 1
    
    return best_centers, best_radius


def _polish(centers, deadline):
    """Final polish with tiny steps to extract precision."""
    best_radius = _compute_radius_fast(centers)
    best_centers = centers.copy()
    
    for it in range(2000):
        if time.time() > deadline:
            break
        
        # Find the current minimum constraint
        wall_dists = np.minimum(
            np.minimum(HALF - centers[:, 0], HALF - centers[:, 1]),
            np.minimum(centers[:, 0] + HALF, centers[:, 1] + HALF)
        )
        min_wall = np.min(wall_dists)
        wall_idx = np.argmin(wall_dists)
        
        # Find closest pair
        n = len(centers)
        min_pair = np.inf
        pair_i, pair_j = 0, 1
        for i in range(n):
            diff = centers[i] - centers
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            dists[i] = np.inf
            idx = np.argmin(dists)
            if dists[idx] < min_pair:
                min_pair = dists[idx]
                pair_i, pair_j = i, idx
        
        min_pair_half = min_pair / 2.0
        current_r = min(min_wall, min_pair_half)
        
        # Try small random perturbations on the constraining points
        candidates = []
        if min_wall <= min_pair_half:
            candidates = [wall_idx]
        else:
            candidates = [pair_i, pair_j]
        
        improved = False
        for idx in candidates:
            c = centers[idx].copy()
            # Try 8 directions with tiny step
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                for s in [1e-6, 5e-6, 1e-5]:
                    new_c = c + np.array([np.cos(angle), np.sin(angle)]) * s
                    new_c[0] = np.clip(new_c[0], -HALF, HALF)
                    new_c[1] = np.clip(new_c[1], -HALF, HALF)
                    centers[idx] = new_c
                    r = _compute_radius_fast(centers)
                    if r > best_radius + 1e-14:
                        best_radius = r
                        best_centers = centers.copy()
                        improved = True
                        break
                    else:
                        centers[idx] = c
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try random small perturbation on a random point
            idx = np.random.randint(n)
            c = centers[idx].copy()
            new_c = c + np.random.randn(2) * 1e-6
            new_c[0] = np.clip(new_c[0], -HALF, HALF)
            new_c[1] = np.clip(new_c[1], -HALF, HALF)
            centers[idx] = new_c
            r = _compute_radius_fast(centers)
            if r > best_radius + 1e-14:
                best_radius = r
                best_centers = centers.copy()
            else:
                centers[idx] = c
    
    return best_centers, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Initialize with grid
    centers = _grid_init(N)
    
    # Main optimization
    centers, r = _optimize(centers, 100, deadline)
    
    # Polish if time remains
    if time.time() < deadline - 5:
        centers, r = _polish(centers, deadline)
    
    # Ensure validity: clip to square
    centers[:, 0] = np.clip(centers[:, 0], -HALF, HALF)
    centers[:, 1] = np.clip(centers[:, 1], -HALF, HALF)
    
    return centers
