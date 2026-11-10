"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _normalize_line(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, x1)
    if dy == 0:
        return (0, 1, y1)
    # Normalize so that dx > 0, or dx == 0 and dy > 0
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    elif dx == 0 and dy < 0:
        dy = -dy
    # Line: dx*x - dy*y = c
    c = dx * x1 - dy * y1
    return (dx, dy, c)

def _count_4_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _normalize_line(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    count = 0
    for key, pts in line_points.items():
        if len(pts) == 4:
            count += 1
    return count

def _no_five_collinear(points):
    """Check that no 5 points are collinear."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _normalize_line(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    for key, pts in line_points.items():
        if len(pts) >= 5:
            return False
    return True

def _validate(points):
    """Check distinctness, bounds, and no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if not _no_five_collinear(points):
        return False
    return True

def _evaluate(points):
    """Return (score, valid) for a configuration."""
    if not _validate(points):
        return (0, False)
    score = _count_4_lines(points)
    return (score, True)

def _random_point():
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _generate_initial():
    """Generate a random initial configuration of 16 distinct points."""
    points = []
    seen = set()
    while len(points) < 16:
        p = _random_point()
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            points.append(p)
    return points

def _local_search(points, max_iter=5000, time_limit=100):
    """Local search by single-point replacement."""
    start_time = time.time()
    best_points = [p[:] for p in points]
    best_score, _ = _evaluate(best_points)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to replace
        idx = random.randint(0, 15)
        old_point = best_points[idx][:]
        
        # Try a few random replacements
        improved = False
        for attempt in range(10):
            new_point = _random_point()
            if new_point == old_point:
                continue
            # Check if new point already exists
            if any(p == new_point for i, p in enumerate(best_points) if i != idx):
                continue
            
            trial = [p[:] for p in best_points]
            trial[idx] = new_point
            
            if not _validate(trial):
                continue
            
            trial_score, _ = _evaluate(trial)
            if trial_score > best_score:
                best_points = trial
                best_score = trial_score
                improved = True
                break
        
        if not improved:
            # Try a small perturbation of the old point
            for attempt in range(10):
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                new_point = [old_point[0] + dx, old_point[1] + dy]
                if new_point == old_point:
                    continue
                if abs(new_point[0]) > 1000 or abs(new_point[1]) > 1000:
                    continue
                if any(p == new_point for i, p in enumerate(best_points) if i != idx):
                    continue
                
                trial = [p[:] for p in best_points]
                trial[idx] = new_point
                
                if not _validate(trial):
                    continue
                
                trial_score, _ = _evaluate(trial)
                if trial_score > best_score:
                    best_points = trial
                    best_score = trial_score
                    improved = True
                    break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with the 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _evaluate(best_points)
    
    # Run local search from the grid
    points, score = _local_search(best_points, max_iter=2000, time_limit=30)
    if score > best_score:
        best_points = points
        best_score = score
    
    # Try several random restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        random_points = _generate_initial()
        points, score = _local_search(random_points, max_iter=1500, time_limit=15)
        if score > best_score:
            best_points = points
            best_score = score
    
    # Final polish with more iterations on the best
    remaining = deadline - time.time()
    if remaining > 10:
        points, score = _local_search(best_points, max_iter=3000, time_limit=min(remaining - 5, 50))
        if score > best_score:
            best_points = points
            best_score = score
    
    return best_points
# EVOLVE-BLOCK-END
