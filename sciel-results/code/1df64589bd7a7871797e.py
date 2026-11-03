"""
Stochastic Hill Climbing with Simulated Annealing for 100 spheres in a cube.

Strategy:
1. Initialize with a jittered grid to ensure a valid starting point.
2. Use Simulated Annealing to maximize the true radius score.
   - The objective is the exact radius r.
   - Perturbations are random Gaussian moves.
   - Acceptance criterion: Metropolis (accept if better, or with probability exp(-delta/T)).
   - Temperature T starts high and decays exponentially.
3. After the annealing phase, perform a "Geometric Polish":
   - Identify active constraints (pairs closer than 2r + eps, walls closer than r + eps).
   - For each active pair, push the points apart along the connecting vector.
   - For each active wall, push the point inward along the normal.
   - Repeat until no improvement or time limit.

This approach avoids the pitfalls of smooth surrogates that may not correlate perfectly with the max-min objective in the final stages, and the stochastic nature helps escape local minima that gradient-based methods (like L-BFGS-B on energy) might get stuck in.
"""

import numpy as np
import time
import random
import math

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
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _grid_init():
    """Generate a 5x5x4 grid with jitter."""
    nx, ny, nz = 5, 5, 4
    # Centers of the grid cells
    # x: 5 points. Spacing 0.2. Range [-0.4, 0.4]
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # Add small jitter to break symmetry
    points += np.random.normal(0, 0.01, (N, 3))
    # Clip to ensure inside cube (with margin)
    points = np.clip(points, -0.45, 0.45)
    return points

def _polish(points, max_iter=500, eps=1e-5):
    """
    Geometric polish: iteratively resolve active constraints.
    1. Calculate r.
    2. Find pairs with dist < 2r + eps. Push them apart.
    3. Find points with margin < r + eps. Push them in.
    4. Repeat.
    """
    pts = points.copy()
    r = _score(pts)
    
    for _ in range(max_iter):
        improved = False
        
        # Calculate distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Active pairs: dist < 2r + eps
        # We want to increase the minimum distance.
        # Find the smallest distance
        min_dist_idx = np.unravel_index(dists.argmin(), dists.shape)
        i, j = min_dist_idx
        
        if dists[i, j] < 2 * r + eps:
            # Push i and j apart
            d = dists[i, j]
            if d < 1e-12:
                # Random push
                pts[i] += np.random.normal(0, 1e-4, 3)
                pts[j] -= np.random.normal(0, 1e-4, 3)
            else:
                vec = pts[i] - pts[j]
                vec /= d
                # Move each by half the deficit
                deficit = (2 * r + eps) - d
                step = deficit / 2.0
                pts[i] += vec * step
                pts[j] -= vec * step
                improved = True
        
        # Active walls: margin < r + eps
        margin = 0.5 - np.abs(pts).max(axis=1)
        min_margin_idx = margin.argmin()
        
        if margin[min_margin_idx] < r + eps:
            p = pts[min_margin_idx]
            # Find the tightest wall
            for k in range(3):
                if abs(p[k]) == np.abs(p).max():
                    # Determine direction to push inward
                    if p[k] > 0:
                        push = -np.sign(p[k])
                    else:
                        push = np.sign(p[k])
                    
                    deficit = (r + eps) - (0.5 - abs(p[k]))
                    if deficit > 0:
                        pts[min_margin_idx, k] += push * deficit
                        improved = True
                    break
        
        if not improved:
            break
            
        # Re-check score occasionally to ensure we aren't degrading
        # But since we are pushing away from constraints, r should not decrease significantly
        # However, pushing one pair might violate another.
        # The SA loop will handle global optimization. This is just a local fix.
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial solution
    points = _grid_init()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Simulated Annealing Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-8
    alpha = 0.995  # Cooling rate
    step_size = 0.05  # Initial step size
    
    # Phase 1: Simulated Annealing
    current_score = best_score
    current_points = best_points.copy()
    
    while time.time() < deadline - 20:
        # Pick a random point to move
        idx = random.randint(0, N - 1)
        
        # Generate a random move
        move = np.random.normal(0, step_size, 3)
        
        new_points = current_points.copy()
        new_points[idx] += move
        
        # Ensure inside cube (hard constraint for validity)
        # If it goes out, reflect or clip? Clipping is safer for validity.
        new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
        
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Metropolis acceptance
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                # Probability of accepting worse move
                # exp(-delta / T)
                # If T is very small, this is 0
                prob = math.exp(-delta / T) if T > 0 else 0
                accept = random.random() < prob
            
            if accept:
                current_points = new_points
                current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Reset step size or restart?
            # Let's just keep going with smaller steps
            step_size *= 0.95
            if step_size < 1e-6:
                step_size = 1e-6

    # Phase 2: Polish the best solution
    if time.time() < deadline - 5:
        polished = _polish(best_points)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
            
        # Try polishing the current annealed state too
        polished2 = _polish(current_points)
        polished2_score = _score(polished2)
        if polished2_score > best_score:
            best_points = polished2
            best_score = polished2_score

    # Final check
    return best_points

# EVOLVE-BLOCK-END
