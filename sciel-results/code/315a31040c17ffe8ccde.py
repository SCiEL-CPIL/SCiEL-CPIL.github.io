"""
Directed Local Search (DLS) + Force-Directed Relaxation for sphere packing.
Strategy:
1. Initialize with a jittered grid.
2. Use a force-directed model (inverse-square repulsion + wall repulsion) to locally
   maximize the minimum distance. This is differentiable and smooth.
3. Use DLS moves (random jitter, swap, 2-opt style re-insertion) to escape local minima.
4. Evaluate score using exact min distance (no soft-min approximation).
"""
import numpy as np
import time
import random

N = 100

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    # Using scipy.spatial.distance for efficiency if possible, but numpy is fine for N=100
    # dist matrix calculation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _compute_forces(points, k_repel=1.0, k_wall=10.0):
    """
    Compute forces on each point.
    - Repulsive force between spheres: F = k_repel * (r_min / dist)^2 * dir
      We want to push them apart. If dist < 2*r_target, push apart.
      Actually, simpler: force proportional to 1/dist^2 pushing away.
    - Wall repulsion: Push away from walls if close.
    """
    forces = np.zeros_like(points)
    n = len(points)
    
    # Pairwise repulsion
    # We want to maximize min dist. 
    # A common heuristic is to use a potential energy U = sum 1/d_ij^2 and minimize it.
    # Gradient of 1/d^2 is -2/d^3 * (r_i - r_j).
    # So force on i is sum_j 2/d^3 * (r_i - r_j) ? No, that minimizes potential.
    # Let's just use a simple repulsive force: F_ij = k * (r_i - r_j) / |r_i - r_j|^3
    # This pushes them apart.
    
    # Vectorized calculation for N=100 is fast enough
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    # Avoid division by zero
    dists[dist < 1e-8] = 1e-8 
    # Force magnitude: k / dist^2. Direction: diff / dist. Total: k * diff / dist^3
    # We only care about forces that increase the minimum distance.
    # But global repulsion is a good proxy.
    inv_dist_cubed = 1.0 / (dists ** 3)
    np.fill_diagonal(inv_dist_cubed, 0.0)
    
    # F_i = sum_j k * (r_i - r_j) / dist^3
    # diff is r_i - r_j
    forces += np.sum(k_repel * diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
    
    # Wall forces
    # If x > 0.5 - eps, push left. If x < -0.5 + eps, push right.
    # Simple spring: F = -k_wall * (pos - target)
    # Target is 0 for center, but we want to keep inside.
    # Let's use a penalty force that grows near the boundary.
    # F_x = -k_wall * (x - 0.5) for x > 0.5 (but we clip anyway).
    # Better: F = -k_wall * (x - (0.5 - margin)) if x > 0.5 - margin.
    # Let's just use a strong spring towards center if near wall.
    # Actually, for packing, we want them to fill the space.
    # So we want them to move away from walls only if they are too close?
    # No, we want to maximize the *minimum* distance to wall.
    # So we want to push them towards the center if they are near the wall?
    # Or rather, we want to distribute them.
    
    # Let's use a force that pushes away from the nearest wall if within a threshold.
    threshold = 0.1
    for dim in range(3):
        # Positives
        close_pos = points[:, dim] > (0.5 - threshold)
        if np.any(close_pos):
            # Push left
            dist_to_wall = 0.5 - points[close_pos, dim]
            # Force magnitude proportional to how close?
            # Let's just use a constant push if inside threshold
            forces[close_pos, dim] -= k_wall
            
        # Negatives
        close_neg = points[:, dim] < (-0.5 + threshold)
        if np.any(close_neg):
            # Push right
            forces[close_neg, dim] += k_wall
            
    return forces

def _relax(points, steps=50, lr=0.005, k_repel=1.0, k_wall=5.0):
    """
    Perform gradient ascent on the minimum distance using force-directed relaxation.
    We approximate the gradient of the min distance by summing repulsive forces.
    """
    pts = points.copy()
    for _ in range(steps):
        forces = _compute_forces(pts, k_repel, k_wall)
        # Normalize forces to prevent explosion
        norm = np.linalg.norm(forces, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        forces = forces / norm
        
        # Update
        pts += lr * forces
        # Clip to bounds
        pts = np.clip(pts, -0.5, 0.5)
        
    return pts

def _dls_step(points, score, move_type):
    """
    Perform a DLS move.
    move_type: 'jitter', 'swap', 'insert'
    """
    n = len(points)
    
    if move_type == 'jitter':
        # Pick a random point and move it slightly
        idx = random.randint(0, n - 1)
        step_size = 0.02
        delta = np.random.normal(0, step_size, 3)
        new_pts = points.copy()
        new_pts[idx] += delta
        new_pts[idx] = np.clip(new_pts[idx], -0.5, 0.5)
        return new_pts

    elif move_type == 'swap':
        # Pick two random points and swap their positions? No, that's useless if they are identical.
        # Instead, pick two points and move them towards each other? Or away?
        # Let's try: pick two points, move them in a random direction by a small amount.
        idx1, idx2 = random.sample(range(n), 2)
        new_pts = points.copy()
        # Move idx1 and idx2 in opposite directions?
        delta = np.random.normal(0, 0.01, 3)
        new_pts[idx1] += delta
        new_pts[idx2] -= delta
        new_pts[idx1] = np.clip(new_pts[idx1], -0.5, 0.5)
        new_pts[idx2] = np.clip(new_pts[idx2], -0.5, 0.5)
        return new_pts

    elif move_type == 'insert':
        # Pick a point, remove it, and re-insert it at a random location?
        # Or pick a point and move it to a random location?
        idx = random.randint(0, n - 1)
        new_pts = points.copy()
        # Move to random location
        new_pts[idx] = np.random.uniform(-0.4, 0.4, 3)
        return new_pts

    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial solution: Jittered grid
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # Add small jitter to break symmetry
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Optimization Loop
    # Alternate between relaxation and DLS moves
    while time.time() < deadline:
        # 1. Relaxation phase
        # Use adaptive learning rate
        current_score = _score(points)
        
        # Try relaxation with different parameters
        # We want to increase the score.
        # Relaxation is a local search.
        relaxed = _relax(points, steps=20, lr=0.01, k_repel=1.0, k_wall=5.0)
        relaxed_score = _score(relaxed)
        
        if relaxed_score > current_score:
            points = relaxed
            current_score = relaxed_score
            if current_score > best_score:
                best_points = points.copy()
                best_score = current_score
        else:
            # If relaxation failed, try DLS moves
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                move_type = random.choice(['jitter', 'swap', 'insert'])
                new_pts = _dls_step(points, current_score, move_type)
                new_score = _score(new_pts)
                
                # Accept if better, or with some probability if worse (Simulated Annealing style)
                # But let's stick to strict improvement for DLS to be safe, 
                # or use a small temperature.
                if new_score > current_score:
                    points = new_pts
                    current_score = new_score
                    if current_score > best_score:
                        best_points = points.copy()
                        best_score = current_score
                else:
                    # Occasionally accept worse moves to escape local minima
                    # Probability decreases as we get closer to deadline
                    if random.random() < 0.1:
                        points = new_pts
                        current_score = new_score
        
        # Periodic re-initialization of a subset?
        # Or just continue.
        
        # Check time
        if time.time() > deadline - 2:
            break

    return best_points

# EVOLVE-BLOCK-END
