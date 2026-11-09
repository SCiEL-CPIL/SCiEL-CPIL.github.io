"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 3:
        return n
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                g = 1
            # Canonical direction vector
            dxn = dx // g
            dyn = dy // g
            # Normalize sign: make first non-zero component positive
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn = -dxn
                dyn = -dyn
            # Canonical line key: (dxn, dyn, c) where c = dyn*x - dxn*y (up to sign handled by direction)
            # Actually, to get a unique line identifier, we need the line itself, not just direction.
            # Line through (x1,y1) with direction (dxn,dyn): 
            # Normal vector is (-dyn, dxn). Line equation: -dyn*x + dxn*y = c
            # c = -dyn*x1 + dxn*y1
            # To make it canonical, we should normalize the normal vector too.
            # But since we already normalized direction, let's just use the point-line form carefully.
            # A better approach: for each pair, compute the line in canonical form.
            # Line: A*x + B*y = C, where A = -dyn, B = dxn, C = -dyn*x1 + dxn*y1
            # Normalize (A, B, C) by gcd and sign.
            A = -dyn
            B = dxn
            C = -dyn * x1 + dxn * y1
            g2 = gcd(abs(A), gcd(abs(B), abs(C)))
            if g2 == 0:
                g2 = 1
            A //= g2
            B //= g2
            C //= g2
            # Sign normalization: make the first non-zero among (A, B) positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary = 0
    for count in line_counts.values():
        if count == 1:  # A line with exactly one pair means exactly 2 points on it
            ordinary += 1
    return ordinary


def _not_all_collinear(points):
    """Check that not all points are collinear."""
    if len(points) < 3:
        return False
    x1, y1 = points[0]
    x2, y2 = points[1]
    for i in range(2, len(points)):
        x3, y3 = points[i]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross != 0:
            return True
    return False


def _evaluate(points):
    """Return the number of ordinary lines, or None if invalid."""
    if len(points) != 12:
        return None
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 12:
        return None
    # Check bounds
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return None
    if not _not_all_collinear(points):
        return None
    return _count_ordinary_lines(points)


def _generate_hexagon_based_points():
    """Generate initial points based on hexagonal symmetry."""
    # Regular hexagon vertices scaled by s
    s = 100
    import math
    # Use integer approximations for hexagon vertices
    # Hexagon: (s,0), (s/2, s*sqrt(3)/2), ... 
    # For integer coords, use a different approach: place points on a grid with hexagonal symmetry
    # Let's use a 4x3 grid perturbed, or use known configurations
    
    # Start with a 3x4 grid as baseline
    points = []
    for x in range(4):
        for y in range(3):
            points.append([x * 100, y * 100])
    return points


def _generate_random_points():
    """Generate 12 random integer points."""
    points = []
    seen = set()
    while len(points) < 12:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            points.append([x, y])
    return points


def _local_search(points, max_iterations=50000, deadline=None):
    """Greedy local search: try moving each point to a nearby integer position."""
    best_score = _evaluate(points)
    if best_score is None:
        return points, best_score
    
    current_points = [list(p) for p in points]
    current_score = best_score
    
    # Precompute some useful data
    n = len(current_points)
    
    for iteration in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        # Try moving each point to a nearby position
        for i in range(n):
            if deadline and time.time() > deadline:
                break
            x, y = current_points[i]
            # Try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    # Check if this position is already taken
                    if any(current_points[j] == [new_x, new_y] for j in range(n) if j != i):
                        continue
                    old_point = current_points[i]
                    current_points[i] = [new_x, new_y]
                    score = _evaluate(current_points)
                    if score is not None and score < current_score:
                        current_score = score
                        improved = True
                        if score < best_score:
                            best_score = score
                            best_points = [list(p) for p in current_points]
                    else:
                        current_points[i] = old_point
        
        if not improved:
            break
    
    return best_points, best_score


def construct():
    """Construct 12 integer points minimizing ordinary lines."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Strategy 1: Start from 3x4 grid and local search
    points1 = _generate_hexagon_based_points()
    pts, score = _local_search(points1, max_iterations=20000, deadline=deadline)
    if score is not None and score < best_score:
        best_score = score
        best_points = pts
    
    # Strategy 2: Random restarts with local search
    random.seed(42)
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        points_r = _generate_random_points()
        pts, score = _local_search(points_r, max_iterations=10000, deadline=deadline)
        if score is not None and score < best_score:
            best_score = score
            best_points = pts
        # If we've found a very good score, do more focused search
        if best_score <= 6:
            # Try more restarts to see if we can do better
            continue
    
    if best_points is None:
        # Fallback to 3x4 grid
        best_points = [[x * 100, y * 100] for x in range(4) for y in range(3)]
    
    return best_points

# EVOLVE-BLOCK-END
