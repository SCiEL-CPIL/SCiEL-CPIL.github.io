"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

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
    """Generate a hexagonal packing of n_circles in [-0.5, 0.5]^2."""
    # Hexagonal grid: rows with alternating offsets
    # For a unit square, we need to find optimal spacing
    # Let's try to fit as many as possible in a hexagonal pattern
    
    # The hexagonal grid has rows with vertical spacing h = sqrt(3)/2 * s
    # and horizontal spacing s, with alternating rows offset by s/2
    
    # We need to find s such that we can fit n_circles
    # For N=200, roughly 14x14 grid
    
    # Let's try different spacings and pick the one that fits best
    best_positions = None
    best_radius = 0
    
    # Try spacings from a reasonable range
    for s in np.linspace(0.03, 0.05, 100):
        h = np.sqrt(3) / 2 * s
        
        # Generate hexagonal grid points
        points = []
        row = 0
        y = -0.5 + h / 2  # Start from bottom
        while y <= 0.5:
            offset = (s / 2) if (row % 2 == 1) else 0
            x = -0.5 + offset
            while x <= 0.5:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append((x, y))
                x += s
            y += h
            row += 1
        
        if len(points) >= n_circles:
            # Take first n_circles
            positions = np.array(points[:n_circles])
            # Check if all are within bounds
            if np.all(np.abs(positions) <= 0.5):
                r = _score(positions)
                if r > best_radius:
                    best_radius = r
                    best_positions = positions
    
    if best_positions is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_positions = grid[:n_circles].copy()
    
    return best_positions

def _local_push_optimize(positions, n_iter=100, step_size=0.01):
    """Local optimization by pushing overlapping circles apart."""
    positions = positions.copy()
    n = len(positions)
    
    for it in range(n_iter):
        # Compute all pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        # Find minimum distance
        min_dist = np.min(dist)
        if min_dist > 0.07:  # 2 * 0.035, roughly 2*r
            break
        
        # Find the pair with minimum distance
        i_min, j_min = np.unravel_index(np.argmin(dist), dist.shape)
        
        # Push them apart
        direction = positions[i_min] - positions[j_min]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / norm
        
        # Push apart by half the overlap
        overlap = 0.073 - min_dist  # 2 * 0.0365
        if overlap > 0:
            positions[i_min] += direction * overlap / 2
            positions[j_min] -= direction * overlap / 2
        
        # Keep within bounds
        positions[:, 0] = np.clip(positions[:, 0], -0.5, 0.5)
        positions[:, 1] = np.clip(positions[:, 1], -0.5, 0.5)
    
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin from 150s limit

    best_score = 0.0
    best_positions = None

    # Try hexagonal grid first
    hex_pos = _generate_hexagonal_grid(N)
    hex_score = _score(hex_pos)
    if hex_score > best_score:
        best_score = hex_score
        best_positions = hex_pos.copy()
    
    # Try square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    square_pos = grid[:N].copy()
    square_score = _score(square_pos)
    if square_score > best_score:
        best_score = square_score
        best_positions = square_pos.copy()

    # Local push optimization
    if best_positions is not None and time.time() < deadline - 5:
        optimized_pos = _local_push_optimize(best_positions, n_iter=50, step_size=0.01)
        opt_score = _score(optimized_pos)
        if opt_score > best_score:
            best_score = opt_score
            best_positions = optimized_pos.copy()
    
    # Try a few random restarts with local optimization
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        
        random_pos = np.random.uniform(-0.45, 0.45, (N, 2))
        rand_score = _score(random_pos)
        
        if rand_score > best_score:
            best_score = rand_score
            best_positions = random_pos.copy()
        
        # Local optimization from random start
        if time.time() < deadline - 10:
            opt_pos = _local_push_optimize(random_pos, n_iter=30, step_size=0.01)
            opt_score = _score(opt_pos)
            if opt_score > best_score:
                best_score = opt_score
                best_positions = opt_pos.copy()

    # Final high-precision polish
    if best_positions is not None and time.time() < deadline - 5:
        # Do more iterations of local push
        polished_pos = _local_push_optimize(best_positions, n_iter=200, step_size=0.005)
        polish_score = _score(polished_pos)
        if polish_score > best_score:
            best_score = polish_score
            best_positions = polished_pos.copy()

    return best_positions
# EVOLVE-BLOCK-END
