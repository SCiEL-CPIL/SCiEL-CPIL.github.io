"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import Counter

# EVOLVE-BLOCK-START
import numpy as np

def _compute_score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
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
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _inbounds(points, limit=30):
    for x, y in points:
        if x < -limit or x > limit or y < -limit or y > limit:
            return False
    return True

def _build_grid(w, h, x0=0, y0=0):
    return [(x0 + x, y0 + y) for x in range(w) for y in range(h)]

def _local_improve(points, max_iters=2000, timeout=45):
    """Local improvement via point moves."""
    start_time = time.time()
    best_points = list(points)
    best_score = _compute_score(best_points)
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Try moving a random point to a random location
        idx = random.randint(0, len(best_points) - 1)
        old_point = best_points[idx]
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        new_point = (new_x, new_y)
        
        # Check if point already exists
        if new_point in best_points:
            continue
        
        best_points[idx] = new_point
        new_score = _compute_score(best_points)
        
        if new_score > best_score:
            best_score = new_score
        else:
            best_points[idx] = old_point
    
    return best_points, best_score

def _generate_candidate(w, h, x0=None, y0=None):
    """Generate a grid candidate."""
    if x0 is None:
        x0 = random.randint(-30, 30 - w + 1)
    if y0 is None:
        y0 = random.randint(-30, 30 - h + 1)
    return _build_grid(w, h, x0, y0)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s safety margin
    
    # Try various grid configurations
    best_points = None
    best_score = 0
    
    # Try rectangular grids that give exactly 30 points
    grid_configs = [
        (6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15),
        (30, 1), (1, 30), (5, 5), (6, 5)
    ]
    
    # Also try some structured non-rectangular patterns
    # Parallelogram-like patterns
    for w, h in grid_configs:
        if time.time() > deadline:
            break
        for _ in range(3):
            if time.time() > deadline:
                break
            points = _generate_candidate(w, h)
            if not _inbounds(points):
                continue
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = list(points)
    
    # Try local improvement from best grid
    if best_points is not None:
        if time.time() < deadline - 10:
            improved_points, improved_score = _local_improve(
                best_points, max_iters=1000, timeout=30
            )
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_points
    
    # Try random starts with local improvement
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            # Generate random 30 points
            points = []
            used = set()
            while len(points) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in used:
                    points.append((x, y))
                    used.add((x, y))
            
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = list(points)
            
            if time.time() < deadline - 5:
                improved_points, improved_score = _local_improve(
                    points, max_iters=500, timeout=15
                )
                if improved_score > best_score:
                    best_score = improved_score
                    best_points = improved_points
    
    if best_points is None:
        # Fallback to seed
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
