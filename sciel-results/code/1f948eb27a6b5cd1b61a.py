"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1
H = np.sqrt(3) / 2.0     # height of the triangle

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])


def _barycentric_to_cartesian(b):
    """Convert barycentric coordinates (u,v,w) to Cartesian."""
    u, v, w = b
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.array([x, y])


def _cartesian_to_barycentric(p):
    """Convert Cartesian point to barycentric coordinates."""
    x, y = p
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.array([u, v, w])


def _is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the triangle."""
    b = _cartesian_to_barycentric(p)
    return all(bi >= -eps for bi in b)


def _clamp_to_triangle(p, eps=1e-6):
    """Clamp point to be inside the triangle."""
    b = _cartesian_to_barycentric(p)
    b = np.maximum(b, 0.0)
    s = b.sum()
    if s < 1e-12:
        return (V0 + V1 + V2) / 3.0
    b = b / s
    return _barycentric_to_cartesian(b)


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                                 - (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
                if area < min_area:
                    min_area = area
    return min_area


def _min_triangle_area_normalized(points):
    """Return min triangle area divided by triangle area."""
    return _min_triangle_area(points) / AREA


def _local_optimize_heuristic(points, time_limit=5.0):
    """
    Local optimization using a heuristic 'push apart' strategy with simulated annealing-like kicks.
    """
    start_time = time.time()
    points = np.array(points, dtype=float)
    best_points = points.copy()
    best_score = _min_triangle_area(points)
    
    current_score = best_score
    step_size = 0.1
    stuck_count = 0
    max_stuck = 50
    min_step = 1e-5
    
    n = len(points)
    
    while time.time() - start_time < time_limit:
        # Find the smallest triangle
        min_area = float('inf')
        min_idx = (0, 1, 2)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1])
                                     - (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
                    if area < min_area:
                        min_area = area
                        min_idx = (i, j, k)
        
        if min_area > current_score:
            current_score = min_area
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
            stuck_count = 0
        else:
            stuck_count += 1
        
        # If stuck for a while, kick the system (annealing)
        if stuck_count > max_stuck:
            # Randomly perturb all points
            for idx in range(n):
                perturbation = np.random.randn(2) * step_size * 0.5
                new_p = points[idx] + perturbation
                if _is_inside_triangle(new_p):
                    points[idx] = new_p
            stuck_count = 0
            step_size = max(step_size * 0.8, min_step)
            continue # Re-evaluate after kick
        
        # Try to move the vertices of the smallest triangle
        i, j, k = min_idx
        centroid = (points[i] + points[j] + points[k]) / 3.0
        
        improved = False
        # Try moving each vertex
        for idx in [i, j, k]:
            # Direction 1: Away from centroid
            direction = points[idx] - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0])
            
            # Try a few step sizes and directions
            for factor in [1.0, 0.5, 0.1]:
                new_p = points[idx] + step_size * factor * direction
                if _is_inside_triangle(new_p):
                    old_p = points[idx]
                    points[idx] = new_p
                    new_score = _min_triangle_area(points)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        points[idx] = old_p
            
            # If not improved, try random directions
            if not improved:
                for _ in range(3):
                    rand_dir = np.random.randn(2)
                    rand_dir = rand_dir / np.linalg.norm(rand_dir)
                    for factor in [1.0, 0.5]:
                        new_p = points[idx] + step_size * factor * rand_dir
                        if _is_inside_triangle(new_p):
                            old_p = points[idx]
                            points[idx] = new_p
                            new_score = _min_triangle_area(points)
                            if new_score > current_score:
                                current_score = new_score
                                improved = True
                                break
                            else:
                                points[idx] = old_p
                        if improved:
                            break
                if improved:
                    break
        
        if not improved:
            # Decay step size
            step_size = max(step_size * 0.95, min_step)
            if step_size <= min_step * 1.01:
                break # Converged
        
    return best_points


def _generate_initial_config(rng, strategy='structured'):
    """Generate an initial configuration of 11 points."""
    if strategy == 'structured':
        points = []
        # Corners (slightly inset)
        points.append(V0 + 0.05 * (V1 - V0))
        points.append(V1 - 0.05 * (V1 - V0))
        points.append(V2 + 0.05 * (V0 - V2))
        
        # Midpoints of edges
        points.append((V0 + V1) / 2.0)
        points.append((V1 + V2) / 2.0)
        points.append((V2 + V0) / 2.0)
        
        # Center
        points.append((V0 + V1 + V2) / 3.0)
        
        # Additional points
        points.append(V0 + 0.3 * (V1 - V0) + 0.1 * (V2 - V0))
        points.append(V0 + 0.1 * (V1 - V0) + 0.3 * (V2 - V0))
        points.append(V1 + 0.1 * (V2 - V1) + 0.1 * (V0 - V1))
        
        return np.array(points)
    elif strategy == 'random':
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95 and u > 0.05 and v > 0.05:
                p = [u + 0.5 * v, H * v]
                points.append(p)
        return np.array(points)
    elif strategy == 'grid':
        points = []
        # Use a 3x3 grid in barycentric coordinates, picking 11 points
        grid_u = [0.1, 0.35, 0.6, 0.85]
        grid_v = [0.1, 0.35, 0.6, 0.85]
        candidates = []
        for u in grid_u:
            for v in grid_v:
                w = 1.0 - u - v
                if w >= -0.01:
                    w = max(w, 0.0)
                    candidates.append(_barycentric_to_cartesian(np.array([u, v, w])))
        
        rng.shuffle(candidates)
        for p in candidates:
            if len(points) >= N:
                break
            points.append(p)
        
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95 and u > 0.05 and v > 0.05:
                points.append([u + 0.5 * v, H * v])
        
        return np.array(points)
    else:
        return _generate_initial_config(rng, 'random')


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = random.Random(42)
    
    best_points = None
    best_score = -float('inf')
    
    strategies = ['structured', 'grid', 'random']
    
    # We will allocate time to multiple restarts
    # Let's say we try 5 restarts per strategy initially, then more if time permits
    total_restarts = 15
    
    for restart in range(total_restarts):
        if time.time() > deadline - 10:
            break
        
        strategy = strategies[restart % len(strategies)]
        points = _generate_initial_config(rng, strategy)
        
        # Allocate time for local optimization
        # Use a fixed time slice per restart, e.g., 5 seconds
        opt_points = _local_optimize_heuristic(points, time_limit=5.0)
        
        score = _min_triangle_area_normalized(opt_points)
        
        if score > best_score:
            best_score = score
            best_points = opt_points.copy()
        
        # If we have lots of time left, do more fine tuning on the best found so far
        # But the heuristic optimizer already runs for 5s, so this might be redundant
        # Let's just do more restarts
    
    # Final fine-tuning on the best solution with a longer time budget
    if best_points is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            points_fine = _local_optimize_heuristic(best_points, time_limit=remaining_time)
            score_fine = _min_triangle_area_normalized(points_fine)
            if score_fine > best_score:
                best_score = score_fine
                best_points = points_fine.copy()
    
    if best_points is not None:
        best_points = np.array([_clamp_to_triangle(p) for p in best_points])
        
    return best_points

# EVOLVE-BLOCK-START
# (The construct function is defined above, but we keep this block for compatibility)
# EVOLVE-BLOCK-END
