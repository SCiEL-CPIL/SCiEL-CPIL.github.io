"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
import time
import math

N = 200

def _compute_radius(pos):
    """Compute the actual packing radius from positions."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Distance to boundary: min over all coordinates of (0.5 - |coord|)
    dist_boundary = 0.5 - np.max(np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = float(np.min(dist_sq))
    min_pair = math.sqrt(min_pair_sq)
    return min(min_pair / 2.0, min_boundary)

def _initial_hex_grid(n):
    """Create initial hexagonal lattice packing."""
    rows = int(np.ceil(np.sqrt(n * 2 / 3)))
    cols = int(np.ceil(n / rows))
    
    points = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            x = float(col)
            y = float(row) * np.sqrt(3) / 2
            if row % 2 == 1:
                x += 0.5
            points.append([x, y])
            count += 1
        if count >= n:
            break
    
    points = np.array(points[:n], dtype=np.float64)
    
    # Center the lattice
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    points -= center
    
    # Scale to fit in [-0.5, 0.5]^2 with margin
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        scale = 0.48 / max_coord
        points *= scale
    
    return points

def _initial_square_grid(n):
    """Create initial square grid packing."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    points = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            x = float(col)
            y = float(row)
            points.append([x, y])
            count += 1
        if count >= n:
            break
    
    points = np.array(points[:n], dtype=np.float64)
    
    # Center the lattice
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    points -= center
    
    # Scale to fit in [-0.5, 0.5]^2 with margin
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        scale = 0.48 / max_coord
        points *= scale
    
    return points

def _initial_random(n, seed=42):
    """Create initial random positions."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 2)).astype(np.float64)

def _vectorized_refine(pos, time_budget=20.0):
    """
    Refine positions using vectorized repulsive forces.
    Minimizes a potential energy that encourages large minimum distances.
    """
    start = time.time()
    pos = pos.copy().astype(np.float64)
    n = len(pos)
    
    # Parameters
    dt = 0.001
    damping = 0.95
    
    vel = np.zeros_like(pos)
    
    for step in range(100000):
        if time.time() - start > time_budget:
            break
            
        # Compute pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force magnitude: 2/d^3
        dist_sq_safe = dist_sq + 1e-12
        inv_d3 = 1.0 / (dist_sq_safe * np.sqrt(dist_sq_safe))
        np.fill_diagonal(inv_d3, 0.0)
        
        # Force on i is sum_j 2 * inv_d3[i,j] * (p_i - p_j)
        forces = 2.0 * np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Wall forces
        x = pos[:, 0]
        y = pos[:, 1]
        
        wall_limit = 0.49
        k_wall = 10.0
        
        fx_wall = np.zeros(n)
        fx_wall[x > wall_limit] = -k_wall * (x[x > wall_limit] - wall_limit)
        fx_wall[x < -wall_limit] = -k_wall * (x[x < -wall_limit] + wall_limit)
        
        fy_wall = np.zeros(n)
        fy_wall[y > wall_limit] = -k_wall * (y[y > wall_limit] - wall_limit)
        fy_wall[y < -wall_limit] = -k_wall * (y[y < -wall_limit] + wall_limit)
        
        forces[:, 0] += fx_wall
        forces[:, 1] += fy_wall
        
        # Update velocity
        vel = damping * vel + dt * forces
        pos = pos + dt * vel
        
        # Hard clip
        pos[:, 0] = np.clip(pos[:, 0], -0.499, 0.499)
        pos[:, 1] = np.clip(pos[:, 1], -0.499, 0.499)
        
        if step % 100 == 0:
            if np.max(np.abs(vel)) < 1e-6:
                break

    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_pos = None
    best_r = 0.0
    
    inits = [
        _initial_hex_grid(N),
        _initial_square_grid(N),
        _initial_random(N, seed=1),
        _initial_random(N, seed=2),
        _initial_random(N, seed=3),
        _initial_random(N, seed=42),
    ]
    
    for pos in inits:
        if time.time() > deadline - 15:
            break
            
        pos = pos.astype(np.float64)
        refined_pos = _vectorized_refine(pos, time_budget=15.0)
        r = _compute_radius(refined_pos)
        
        if r > best_r:
            best_r = r
            best_pos = refined_pos.copy()
            
        if time.time() > deadline - 5:
            break
            
    if best_pos is None:
        return _initial_hex_grid(N)
        
    return best_pos
