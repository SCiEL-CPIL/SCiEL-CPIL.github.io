"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(positions):
    """Compute the minimum radius given positions in [-0.5, 0.5]^2."""
    n = len(positions)
    if n == 0:
        return 0.0
    # Distance to boundary
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
    """Generate a hexagonal grid of n_circles points centered in [-0.5, 0.5]^2."""
    points = []
    # Hexagonal lattice: row spacing = sqrt(3)/2 * d, column spacing = d
    # Alternate rows offset by d/2
    # We'll generate more than needed and trim
    max_rows = int(np.ceil(np.sqrt(n_circles * 2 / np.sqrt(3)))) + 2
    d = 1.0  # Will scale later
    for row in range(-max_rows, max_rows + 1):
        offset = (row % 2) * 0.5 * d
        for col in range(-max_rows, max_rows + 1):
            x = col * d + offset
            y = row * np.sqrt(3) / 2 * d
            points.append([x, y])
    
    points = np.array(points)
    # Sort by distance from origin to get centered subset
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)[:n_circles]
    points = points[idx]
    
    # Scale to fit in [-0.5, 0.5]^2
    max_coord = np.max(np.abs(points))
    if max_coord > 0.5:
        points = points * (0.5 / max_coord)
    
    return points

def _local_push_optimize(positions, max_iters=1000, tol=1e-10):
    """Iteratively push apart the closest pairs to increase minimum distance."""
    positions = positions.copy()
    n = len(positions)
    
    for it in range(max_iters):
        # Find closest pair
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        min_idx = np.argmin(dist)
        i, j = np.unravel_index(min_idx, (n, n))
        if i >= j:
            i, j = j, i
        
        min_dist = dist[i, j]
        if min_dist < tol:
            break
        
        # Compute push direction
        direction = positions[i] - positions[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-15:
            direction = np.array([1.0, 0.0])
            norm = 1.0
        else:
            direction = direction / norm
        
        # Push amount: half the deficit to reach target, but limited
        # We want to increase min_dist, so push by a small fraction
        push = 0.01 * (1.0 - min_dist / 1.0)  # Scaled push
        push = max(push, 0.001)
        
        positions[i] += direction * push
        positions[j] -= direction * push
        
        # Clamp to bounds
        positions = np.clip(positions, -0.5, 0.5)
    
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_positions = None
    
    # Generate hexagonal grid initialization
    hex_pos = _generate_hexagonal_grid(N)
    curr_score = _score(hex_pos)
    if curr_score > best_score:
        best_score = curr_score
        best_positions = hex_pos.copy()
    
    # Also try square grid as backup
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    sq_pos = grid[:N].copy()
    sq_score = _score(sq_pos)
    if sq_score > best_score:
        best_score = sq_score
        best_positions = sq_pos.copy()
    
    # Define bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Local push optimization from hexagonal grid
    if time.time() < deadline - 30:
        try:
            pushed = _local_push_optimize(hex_pos, max_iters=500)
            push_score = _score(pushed)
            if push_score > best_score:
                best_score = push_score
                best_positions = pushed.copy()
        except Exception:
            pass
    
    # L-BFGS-B refinement from best position
    if time.time() < deadline - 20:
        try:
            def _objective(pos_flat):
                pos = pos_flat.reshape(-1, 2)
                out_of_bounds = np.any(pos > 0.5) or np.any(pos < -0.5)
                if out_of_bounds:
                    return 1e6
                return -_score(pos)
            
            res = minimize(
                _objective,
                best_positions.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_pos = res.x.reshape(-1, 2)
            new_score = _score(new_pos)
            if new_score > best_score:
                best_score = new_score
                best_positions = new_pos.copy()
        except Exception:
            pass
    
    # Second round of local push from refined position
    if time.time() < deadline - 10:
        try:
            pushed2 = _local_push_optimize(best_positions, max_iters=300)
            push_score2 = _score(pushed2)
            if push_score2 > best_score:
                best_score = push_score2
                best_positions = pushed2.copy()
        except Exception:
            pass
    
    # Final L-BFGS-B polish
    if time.time() < deadline - 5:
        try:
            def _objective2(pos_flat):
                pos = pos_flat.reshape(-1, 2)
                out_of_bounds = np.any(pos > 0.5) or np.any(pos < -0.5)
                if out_of_bounds:
                    return 1e6
                return -_score(pos)
            
            res2 = minimize(
                _objective2,
                best_positions.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            
            new_pos2 = res2.x.reshape(-1, 2)
            new_score2 = _score(new_pos2)
            if new_score2 > best_score:
                best_score = new_score2
                best_positions = new_pos2.copy()
        except Exception:
            pass
    
    return best_positions

# EVOLVE-BLOCK-END
