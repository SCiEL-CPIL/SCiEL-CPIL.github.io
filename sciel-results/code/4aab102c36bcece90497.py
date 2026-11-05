"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def _compute_areas(points):
    """Compute all doubled triangle areas (absolute determinants) for a set of points."""
    n = len(points)
    if n < 3:
        return []
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs(x1*(y2 - y3) - x2*(y1 - y3) + x3*(y1 - y2))
                if det > 0:
                    areas.append(det)
    return areas

def _score_config(points):
    """Return the maximum multiplicity of any positive doubled area."""
    areas = _compute_areas(points)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _score_target(points, target_area):
    """Count how many triples have exactly target_area as doubled area."""
    n = len(points)
    count = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs(x1*(y2 - y3) - x2*(y1 - y3) + x3*(y1 - y2))
                if det == target_area:
                    count += 1
    return count

def _local_search(points, target_area, max_iter=5000, time_limit=30):
    """Local search to maximize count of triples with target_area."""
    start_time = time.time()
    best_points = list(points)
    best_count = _score_target(best_points, target_area)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        idx = random.randint(0, len(best_points) - 1)
        old_x, old_y = best_points[idx]
        
        # Try random new positions
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        
        # Check distinctness
        is_distinct = True
        for i, (px, py) in enumerate(best_points):
            if i != idx and (px, py) == (new_x, new_y):
                is_distinct = False
                break
        
        if not is_distinct:
            continue
        
        # Try the move
        best_points[idx] = (new_x, new_y)
        new_count = _score_target(best_points, target_area)
        
        if new_count >= best_count:
            best_count = new_count
        else:
            # Revert
            best_points[idx] = (old_x, old_y)
    
    return best_points, best_count

def _initial_grid_points():
    """Generate initial grid of 30 points."""
    points = []
    for x in range(6):
        for y in range(5):
            points.append((x, y))
    return points

def construct():
    deadline = time.time() + 45  # Safety margin before 60s limit
    
    # Start with grid configuration
    points = _initial_grid_points()
    best_score = _score_config(points)
    best_points = list(points)
    
    # Determine good target areas from initial config
    areas = _compute_areas(points)
    if areas:
        counts = Counter(areas)
        # Sort by frequency, try top candidates
        target_areas = [a for a, c in counts.most_common(10)]
    else:
        target_areas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # For each target area, do local search
    for target in target_areas:
        if time.time() > deadline:
            break
        
        # Run local search for this target
        improved_points, improved_count = _local_search(
            best_points, target, max_iter=3000, time_limit=10
        )
        
        if time.time() > deadline:
            break
        
        # Evaluate the actual max multiplicity for this config
        actual_score = _score_config(improved_points)
        if actual_score > best_score:
            best_score = actual_score
            best_points = improved_points
    
    # Final check
    if time.time() > deadline:
        return best_points
    
    return best_points
# EVOLVE-BLOCK-END
