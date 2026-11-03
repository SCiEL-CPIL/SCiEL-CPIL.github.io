"""
Strategy: Stochastic local search (Simulated Annealing) with spatial hashing.
Instead of global smooth optimization, we use random perturbations of individual
circle positions accepted/rejected based on a temperature schedule. A spatial
hash grid makes neighbor queries efficient, allowing many iterations within the
time budget. This can escape local minima that trap gradient-based methods.
"""
import numpy as np
import time
import math
import random

N = 300
DEADLINE = 145.0  # Safety margin from 150s limit

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return first N."""
    best_points = None
    for s in [0.10, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.115, 0.12]:
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None or len(pts) > len(best_points):
                best_points = pts
                if len(pts) == N:
                    break
    
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback: very fine grid
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]

def _score(points):
    """Compute the maximum radius r for given centers."""
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _build_spatial_hash(points, cell_size):
    """Build a spatial hash grid for efficient neighbor queries."""
    grid = {}
    for i, (x, y) in enumerate(points):
        cx = int(math.floor(x / cell_size))
        cy = int(math.floor(y / cell_size))
        key = (cx, cy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid

def _get_neighbors(grid, x, y, cell_size):
    """Get indices of points in nearby cells."""
    cx = int(math.floor(x / cell_size))
    cy = int(math.floor(y / cell_size))
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            key = (cx + dx, cy + dy)
            if key in grid:
                neighbors.extend(grid[key])
    return neighbors

def _local_score_delta(points, i, new_pos, grid, cell_size, current_min_dist):
    """
    Compute the change in minimum distance if point i is moved to new_pos.
    Returns (new_min_dist, delta) where delta = new_min_dist - current_min_dist.
    Only considers neighbors within a certain range.
    """
    x_new, y_new = new_pos
    r_new = math.sqrt(x_new * x_new + y_new * y_new)
    boundary_new = 1.0 - r_new
    
    # Check boundary constraint
    if boundary_new < 0:
        return 0, -1e9  # Invalid, large negative delta
    
    # Get potential neighbors
    neighbor_indices = _get_neighbors(grid, x_new, y_new, cell_size)
    # Also check the old position's neighbors
    x_old, y_old = points[i]
    old_neighbors = _get_neighbors(grid, x_old, y_old, cell_size)
    all_candidates = set(neighbor_indices) | set(old_neighbors)
    all_candidates.discard(i)
    
    min_pair_new = float('inf')
    for j in all_candidates:
        if j == i:
            continue
        dx = x_new - points[j, 0]
        dy = y_new - points[j, 1]
        d = math.sqrt(dx * dx + dy * dy)
        if d < min_pair_new:
            min_pair_new = d
    
    # The new minimum distance is the min of boundary and half the closest pair
    new_min = min(boundary_new, min_pair_new / 2.0)
    delta = new_min - current_min_dist
    
    return new_min, delta

def _anneal(points, time_limit, initial_temp=0.01, cooling=0.999):
    """
    Run simulated annealing to improve the packing.
    """
    start_time = time.time()
    points = points.copy()
    best_points = points.copy()
    best_score = _score(points)
    
    # Estimate a reasonable cell size
    current_min_dist = 2 * best_score
    cell_size = max(current_min_dist * 2, 0.05)
    
    temp = initial_temp
    iteration = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Pick a random circle to perturb
        i = random.randint(0, N - 1)
        
        # Generate a random perturbation
        # Scale perturbation with temperature
        scale = temp * 0.5
        dx = random.gauss(0, scale)
        dy = random.gauss(0, scale)
        
        new_pos = (points[i, 0] + dx, points[i, 1] + dy)
        
        # Check if new position is inside the unit circle
        r_new = math.sqrt(new_pos[0] * new_pos[0] + new_pos[1] * new_pos[1])
        if r_new > 1.0:
            # Reflect or reject
            continue
        
        # Build spatial hash (could be cached but rebuilding is simple)
        grid = _build_spatial_hash(points, cell_size)
        
        # Compute delta
        new_min, delta = _local_score_delta(points, i, new_pos, grid, cell_size, best_score)
        
        # Accept/reject
        if delta > 0 or random.random() < math.exp(delta / temp):
            # Accept the move
            points[i, 0] = new_pos[0]
            points[i, 1] = new_pos[1]
            
            # Update best
            current_score = _score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                # Update cell size based on new min distance
                current_min_dist = 2 * best_score
                cell_size = max(current_min_dist * 2, 0.05)
        
        # Cool down
        temp *= cooling
        
        # If temperature is too low, reheat occasionally
        if temp < 0.001:
            temp = initial_temp * 0.5
    
    return best_points, best_score

def construct():
    start_time = time.time()
    
    # Generate initial hexagonal lattice
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        # Emergency fallback
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * np.cos(angle)
            points[i, 1] = r * np.sin(angle)
    
    best_points = points.copy()
    best_score = _score(points)
    
    np.random.seed(42)
    random.seed(42)
    
    # Multi-restart strategy
    max_restarts = 8
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 5.0  # 5s safety margin
        if remaining < 15.0:
            break
        
        if restart == 0:
            current = points.copy()
            time_limit = min(remaining * 0.4, 25.0)
        else:
            # Perturb from best found
            magnitude = 0.015 * (1 + restart * 0.5)
            current = best_points + np.random.normal(0, magnitude, size=best_points.shape)
            # Clamp to disk
            r = np.hypot(current[:, 0], current[:, 1])
            mask = r > 0.98
            if np.any(mask):
                current[mask] = current[mask] / r[mask, np.newaxis] * 0.98
            time_limit = min(remaining * 0.35, 20.0)
        
        optimized, score = _anneal(current, time_limit)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
        
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break
    
    # Final check: ensure validity
    r = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        best_points[mask] = best_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
    
    # Ensure no coincident points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                    best_points[j, 0] += 1e-6
    
    return best_points
