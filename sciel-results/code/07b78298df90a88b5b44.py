"""Stochastic local search packer for 100 equal spheres in a unit cube.

Structural change vs. seed: instead of gradient-based L-BFGS-B on a soft-min
surrogate, we use simulated annealing with a targeted "repair" move that
identifies the most-overlapping sphere and pushes it away from its nearest
neighbours / walls, plus small random jitter. This directly optimizes the true
radius (min of pairwise distances / 2 and wall margins) and can escape local
minima via temperature-controlled acceptance of worse moves.
"""
import numpy as np
import time
import random

N = 100
HALF = 0.5

def _radius(pts):
    """Compute the maximal common radius for the given centres."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Wall margins
    margin = HALF - np.abs(pts).max(axis=1)
    r_wall = margin.min()
    if r_wall <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    return min(r_pair, r_wall)

def _worst_overlap_index(pts, r):
    """Return index of sphere most violating the radius constraint.
    
    A sphere violates if:
    - It has a neighbour closer than 2*r, OR
    - It is too close to a wall (margin < r)
    
    We score each sphere by how badly it violates, and return the worst.
    If no violation, return -1.
    """
    # For efficiency, we compute per-sphere minimum distance to others
    # and per-sphere minimum wall margin.
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # For each sphere i, the tightest pairwise constraint is min_j dists[i,j] / 2
    min_pair_r = dists.min(axis=1) / 2.0
    
    # Wall margins per sphere
    margin = HALF - np.abs(pts).max(axis=1)
    min_wall_r = margin
    
    # Effective radius for sphere i is min(min_pair_r[i], min_wall_r[i])
    # Violation is r - effective_radius[i]
    effective = np.minimum(min_pair_r, min_wall_r)
    violation = r - effective
    
    # Find worst violator
    worst_idx = int(np.argmax(violation))
    if violation[worst_idx] > 1e-12:
        return worst_idx
    return -1

def _push_out(pts, idx, r, strength=1.0):
    """Push sphere idx away from its nearest neighbours and walls.
    
    Returns a new position for sphere idx.
    """
    p = pts[idx].copy()
    
    # Find nearest neighbours
    dists = np.sqrt(((pts - p) ** 2).sum(axis=1))
    dists[idx] = np.inf
    # Get 5 nearest neighbours
    k = min(5, N - 1)
    nn_indices = np.argpartition(dists, k)[:k]
    
    # Compute repulsion force from neighbours
    repulsion = np.zeros(3)
    for j in nn_indices:
        d = p - pts[j]
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            # Random direction
            d = np.random.normal(0, 1, 3)
            dist = np.linalg.norm(d)
        # Repel if too close
        if dist < 2 * r:
            repulsion += (d / dist) * (2 * r - dist)
    
    # Wall repulsion
    wall_force = np.zeros(3)
    for axis in range(3):
        if p[axis] > HALF - r:
            wall_force[axis] = -(p[axis] - (HALF - r))
        elif p[axis] < -(HALF - r):
            wall_force[axis] = -((p[axis] + (HALF - r)))
    
    total_force = repulsion + wall_force
    new_p = p + strength * total_force
    
    # Clip to cube
    new_p = np.clip(new_p, -HALF, HALF)
    return new_p

def _simulated_annealing(points, t_max=0.02, t_min=0.0005, alpha=0.995, max_iter=2000):
    """Simulated annealing with repair moves."""
    current_points = points.copy()
    current_r = _radius(current_points)
    best_points = current_points.copy()
    best_r = current_r
    
    t = t_max
    for i in range(max_iter):
        if t < t_min:
            break
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.7:
            # Repair move: push out worst-overlapping sphere
            worst_idx = _worst_overlap_index(current_points, current_r)
            if worst_idx >= 0:
                strength = 0.5 * t + 0.01
                new_p = _push_out(current_points, worst_idx, current_r, strength)
                new_points = current_points.copy()
                new_points[worst_idx] = new_p
            else:
                # No overlap, just jitter a random sphere
                idx = random.randint(0, N - 1)
                new_points = current_points.copy()
                new_points[idx] += np.random.normal(0, 0.01 * t, 3)
                new_points[idx] = np.clip(new_points[idx], -HALF, HALF)
        else:
            # Pure jitter move
            idx = random.randint(0, N - 1)
            new_points = current_points.copy()
            new_points[idx] += np.random.normal(0, 0.02 * t, 3)
            new_points[idx] = np.clip(new_points[idx], -HALF, HALF)
        
        new_r = _radius(new_points)
        
        if new_r >= current_r:
            current_points = new_points
            current_r = new_r
            if current_r > best_r:
                best_points = current_points.copy()
                best_r = current_r
        else:
            delta = current_r - new_r
            accept_prob = np.exp(-delta / max(t, 1e-10))
            if random.random() < accept_prob:
                current_points = new_points
                current_r = new_r
        
        t *= alpha
    
    return best_points

def _grid_init():
    """Grid-based initialisation: 5x5x4 grid."""
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-HALF + 0.08, HALF - 0.08, nx)
    ys = np.linspace(-HALF + 0.08, HALF - 0.08, ny)
    zs = np.linspace(-HALF + 0.1, HALF - 0.1, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def _random_init():
    """Random initialisation inside the cube."""
    return np.random.uniform(-HALF + 0.05, HALF - 0.05, (N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial solutions
    best_points = _grid_init()
    best_r = _radius(best_points)
    
    # Run multiple SA rounds with different initializations
    n_rounds = 0
    while time.time() < deadline - 10:
        n_rounds += 1
        
        # Choose initialization
        if n_rounds % 3 == 1:
            init_pts = _grid_init() + np.random.normal(0, 0.01, (N, 3))
            init_pts = np.clip(init_pts, -HALF, HALF)
        elif n_rounds % 3 == 2:
            init_pts = _random_init()
        else:
            init_pts = best_points + np.random.normal(0, 0.02, (N, 3))
            init_pts = np.clip(init_pts, -HALF, HALF)
        
        # Run SA with varying parameters
        t_max = 0.02 * (1.0 + 0.5 * random.random())
        t_min = 0.0005
        alpha = 0.995
        max_iter = 2000
        
        sa_points = _simulated_annealing(init_pts, t_max=t_max, t_min=t_min,
                                          alpha=alpha, max_iter=max_iter)
        sa_r = _radius(sa_points)
        
        if sa_r > best_r:
            best_points = sa_points
            best_r = sa_r
    
    # Final polish: short SA with low temperature
    if time.time() < deadline - 2:
        polished = _simulated_annealing(best_points, t_max=0.005, t_min=0.0001,
                                         alpha=0.995, max_iter=1000)
        polished_r = _radius(polished)
        if polished_r > best_r:
            best_points = polished
            best_r = polished_r
    
    return best_points

# EVOLVE-BLOCK-END
