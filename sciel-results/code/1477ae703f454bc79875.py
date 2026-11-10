"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter
import random
import numpy as np

def _doubled_areas(points):
    """Compute all non-zero doubled areas for all triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if det > 0:
                    areas.append(det)
    return areas

def _score(points):
    """Return the maximum multiplicity of any single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _random_points_in_range(n, lo, hi):
    """Generate n distinct random integer points in [lo, hi]^2."""
    points = set()
    while len(points) < n:
        x = random.randint(lo, hi)
        y = random.randint(lo, hi)
        points.add((x, y))
    return list(points)

def _local_search(points, max_iters=50, timeout=None):
    """Local search: try moving one point at a time to improve score."""
    if timeout is None:
        timeout = time.time() + 50
    n = len(points)
    points = list(points)
    best_score = _score(points)
    best_points = points[:]
    
    for it in range(max_iters):
        if time.time() > timeout:
            break
        
        # Try moving each point to a random nearby location
        for _ in range(10):  # 10 random moves per iteration
            if time.time() > timeout:
                break
            
            idx = random.randint(0, n-1)
            ox, oy = points[idx]
            
            # Try a random new position in a local neighborhood
            nx = ox + random.randint(-5, 5)
            ny = oy + random.randint(-5, 5)
            
            # Clamp to range
            nx = max(-30, min(30, nx))
            ny = max(-30, min(30, ny))
            
            # Skip if same point
            if (nx, ny) == (ox, oy):
                continue
            
            # Skip if point already exists
            if (nx, ny) in points:
                continue
            
            # Try the move
            points[idx] = (nx, ny)
            new_score = _score(points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = points[:]
            else:
                # Revert
                points[idx] = (ox, oy)
    
    return best_points, best_score

def construct():
    """Construct 30 points maximizing the max multiplicity of a doubled area."""
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 30
    best_points = None
    best_score = 0
    
    # Start with several grid configurations with different spacings
    grid_configs = [
        [(x*sx, y*sy) for x in range(6) for y in range(5)] for sx in [1,2,3,4,5,6,7,8,9,10] for sy in [1,2,3,4,5,6,7,8,9,10]
        if -30 <= (5*sx) <= 30 and -30 <= (4*sy) <= 30
    ]
    
    for grid in grid_configs:
        if time.time() > deadline:
            break
        score = _score(grid)
        if score > best_score:
            best_score = score
            best_points = grid[:]
    
    # Also try some shifted grids
    for dx in range(-10, 11, 2):
        for dy in range(-10, 11, 2):
            if time.time() > deadline:
                break
            grid = [(x + dx, y + dy) for x in range(6) for y in range(5)]
            score = _score(grid)
            if score > best_score:
                best_score = score
                best_points = grid[:]
    
    # Local search from the best configuration
    if best_points is not None:
        searched_points, searched_score = _local_search(
            best_points, max_iters=200, timeout=deadline
        )
        if searched_score > best_score:
            best_points = searched_points
            best_score = searched_score
    
    # Try a few random restarts
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        random_points = _random_points_in_range(n, -30, 30)
        init_score = _score(random_points)
        
        if init_score > best_score:
            best_points = random_points[:]
            best_score = init_score
        
        searched_points, searched_score = _local_search(
            random_points, max_iters=50, 
            timeout=deadline
        )
        
        if searched_score > best_score:
            best_points = searched_points
            best_score = searched_score
    
    return best_points
# EVOLVE-BLOCK-END
