import time
import random
import itertools
from collections import defaultdict

# Deadline guard
_DEADLINE = time.time() + 140  # 140s out of 150s budget

def _line_key(p1, p2):
    """Return normalized line key (a, b, c) for line ax + by + c = 0."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    # Normalize
    g = _gcd(abs(a), abs(b))
    if g:
        a //= g
        b //= g
        c //= g
    # Make first nonzero positive
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_exact4(points):
    """Count lines containing exactly 4 of the given points."""
    pts = set(tuple(p) for p in points)
    if len(pts) != len(points):
        return -1  # invalid: duplicates
    # Map each line to how many of our points lie on it
    line_counts = defaultdict(int)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] += 1
    # For each line, count how many of our points are on it
    # We need to count distinct points per line
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_points[key].add(tuple(points[i]))
            line_points[key].add(tuple(points[j]))
    
    count4 = 0
    count5plus = 0
    for key, ptset in line_points.items():
        cnt = len(ptset)
        if cnt == 4:
            count4 += 1
        elif cnt >= 5:
            count5plus += 1
    if count5plus > 0:
        return -1  # invalid: has 5+ collinear
    return count4

def _is_valid(points):
    """Check no 5 collinear and all distinct."""
    pts = set(tuple(p) for p in points)
    if len(pts) != len(points):
        return False
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_points[key].add(tuple(points[i]))
            line_points[key].add(tuple(points[j]))
    for key, ptset in line_points.items():
        if len(ptset) >= 5:
            return False
    return True

def _score(points):
    """Return number of exact-4 lines, or -1 if invalid."""
    return _count_exact4(points)

def construct():
    start = time.time()
    
    # Start with the 4x4 grid baseline
    base = [[x, y] for x in range(4) for y in range(4)]
    best_score = _score(base)
    best_points = [p[:] for p in base]
    
    if best_score < 0:
        best_score = 0
    
    random.seed(42)
    
    # Local search: try swapping one point at a time
    # Grid range for new points
    GRID = 12
    
    def _try_swap(points, idx, new_pt):
        """Try replacing points[idx] with new_pt, return (score, new_points) or None."""
        new_pts = [p[:] for p in points]
        new_pts[idx] = new_pt
        if len(set(tuple(p) for p in new_pts)) != 16:
            return None
        s = _score(new_pts)
        if s < 0:
            return None
        return (s, new_pts)
    
    # Iterative improvement with random restarts
    max_iterations = 500000
    
    for iteration in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        
        # Get current best
        cur_pts = best_points
        cur_score = best_score
        
        # Try all single swaps
        improved = False
        for idx in range(16):
            if time.time() > _DEADLINE:
                break
            old_pt = cur_pts[idx]
            # Try random new positions
            for _ in range(20):
                if time.time() > _DEADLINE:
                    break
                nx = random.randint(-GRID, GRID)
                ny = random.randint(-GRID, GRID)
                new_pt = [nx, ny]
                result = _try_swap(cur_pts, idx, new_pt)
                if result is None:
                    continue
                s, new_pts = result
                if s > best_score:
                    best_score = s
                    best_points = new_pts
                    cur_pts = new_pts
                    cur_score = s
                    improved = True
                    break
            if improved:
                break
        
        # Occasionally do a random restart from a perturbed base
        if iteration % 5000 == 4999:
            perturbed = [p[:] for p in base]
            for _ in range(random.randint(2, 6)):
                idx = random.randint(0, 15)
                perturbed[idx] = [random.randint(-GRID, GRID), random.randint(-GRID, GRID)]
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_points = perturbed
    
    # Final validation
    if not _is_valid(best_points):
        return base
    
    return best_points
