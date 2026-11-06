"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    """Check if n is a perfect square (n is an integer)."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist(p1, p2):
    """Squared distance between two points (as rational)."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _dist_is_rational(p1, p2):
    """Check if distance between two points is rational."""
    d2 = _sq_dist(p1, p2)
    return _is_square(d2)

def _no_three_collinear(points):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points):
    """Check no four points are concyclic using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    x1, y1 = p1[0], p1[1]
                    x2, y2 = p2[0], p2[1]
                    x3, y3 = p3[0], p3[1]
                    x4, y4 = p4[0], p4[1]
                    
                    # Compute determinant of the 4x4 matrix for concyclicity
                    # | x^2+y^2  x  y  1 |
                    # | x1^2+y1^2 x1 y1 1 |
                    # | x2^2+y2^2 x2 y2 1 |
                    # | x3^2+y3^2 x3 y3 1 |
                    # | x4^2+y4^2 x4 y4 1 |
                    # Actually, we need a 4x4 determinant with 4 points
                    
                    # Use the standard formula: four points are concyclic iff
                    # the determinant of the following 4x4 matrix is zero:
                    # | x1^2+y1^2  x1  y1  1 |
                    # | x2^2+y2^2  x2  y2  1 |
                    # | x3^2+y3^2  x3  y3  1 |
                    # | x4^2+y4^2  x4  y4  1 |
                    
                    d1 = x1*x1 + y1*y1
                    d2 = x2*x2 + y2*y2
                    d3 = x3*x3 + y3*y3
                    d4 = x4*x4 + y4*y4
                    
                    # Compute 4x4 determinant using cofactor expansion
                    # det = d1*(x2*y3 - x3*y2 - x2*y4 + x4*y2 + x3*y4 - x4*y3)
                    #     - x1*(d2*y3 - d3*y2 - d2*y4 + d4*y2 + d3*y4 - d4*y3)
                    #     + y1*(d2*x3 - d3*x2 - d2*x4 + d4*x2 + d3*x4 - d4*x3)
                    #     - 1*(d2*x3*y4 - d3*x2*y4 - d2*x4*y3 + d4*x2*y3 + d3*x4*y2 - d4*x3*y2)
                    
                    det = (
                        d1 * (x2*y3 - x3*y2 - x2*y4 + x4*y2 + x3*y4 - x4*y3)
                        - x1 * (d2*y3 - d3*y2 - d2*y4 + d4*y2 + d3*y4 - d4*y3)
                        + y1 * (d2*x3 - d3*x2 - d2*x4 + d4*x2 + d3*x4 - d4*x3)
                        - (d2*x3*y4 - d3*x2*y4 - d2*x4*y3 + d4*x2*y3 + d3*x4*y2 - d4*x3*y2)
                    )
                    
                    if det == 0:
                        return False
    return True

def _validate(points):
    """Validate a set of points: all pairwise distances rational, no three collinear, no four concyclic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_is_rational(points[i], points[j]):
                return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points):
        return False
    return True

def _can_add_point(points, new_point):
    """Check if new_point can be added to points."""
    n = len(points)
    for i in range(n):
        if not _dist_is_rational(points[i], new_point):
            return False
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            cross = (p2[0] - p1[0]) * (new_point[1] - p1[1]) - (p2[1] - p1[1]) * (new_point[0] - p1[0])
            if cross == 0:
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                x1, y1 = p1[0], p1[1]
                x2, y2 = p2[0], p2[1]
                x3, y3 = p3[0], p3[1]
                x4, y4 = new_point[0], new_point[1]
                
                d1 = x1*x1 + y1*y1
                d2 = x2*x2 + y2*y2
                d3 = x3*x3 + y3*y3
                d4 = x4*x4 + y4*y4
                
                det = (
                    d1 * (x2*y3 - x3*y2 - x2*y4 + x4*y2 + x3*y4 - x4*y3)
                    - x1 * (d2*y3 - d3*y2 - d2*y4 + d4*y2 + d3*y4 - d4*y3)
                    + y1 * (d2*x3 - d3*x2 - d2*x4 + d4*x2 + d3*x4 - d4*x3)
                    - (d2*x3*y4 - d3*x2*y4 - d2*x4*y3 + d4*x2*y3 + d3*x4*y2 - d4*x3*y2)
                )
                
                if det == 0:
                    return False
    return True

def _generate_candidates(center, radius, n_candidates=200):
    """Generate candidate points near center with integer coordinates."""
    candidates = set()
    for _ in range(n_candidates):
        x = center[0] + random.randint(-radius, radius)
        y = center[1] + random.randint(-radius, radius)
        candidates.add((x, y))
    
    for dx in range(-radius, radius + 1, max(1, radius // 10)):
        for dy in range(-radius, radius + 1, max(1, radius // 10)):
            candidates.add((center[0] + dx, center[1] + dy))
    
    return list(candidates)

def _local_search(initial_points, max_time=10.0):
    """Local search to grow the point set."""
    points = list(initial_points)
    start_time = time.time()
    
    best = len(points)
    best_points = list(points)
    
    while time.time() - start_time < max_time:
        added = False
        for _ in range(50):
            center = random.choice(points)
            candidates = _generate_candidates(center, 50, 100)
            
            for cand in random.sample(candidates, min(20, len(candidates))):
                if cand in points:
                    continue
                if _can_add_point(points, cand):
                    points.append(cand)
                    best = max(best, len(points))
                    best_points = list(points)
                    added = True
                    break
            if added:
                break
        
        if not added:
            if len(points) > 3:
                remove_idx = random.randint(0, len(points) - 1)
                removed = points.pop(remove_idx)
                for _ in range(30):
                    center = random.choice(points)
                    candidates = _generate_candidates(center, 80, 50)
                    for cand in random.sample(candidates, min(15, len(candidates))):
                        if cand in points:
                            continue
                        if _can_add_point(points, cand):
                            points.append(cand)
                            best = max(best, len(points))
                            best_points = list(points)
                            break
                if len(points) <= best:
                    points.append(removed)
    
    return best_points

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Known 5-point integral point sets
    base_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (56, 33), (16, 63), (0, 33), (-16, 63)],
        [(0, 0), (3, 4), (5, 0), (0, 5), (-3, 4)],
        [(0, 0), (12, 5), (5, 12), (0, 13), (-7, 24)],
    ]
    
    best_points = None
    best_len = 0
    
    for base in base_sets:
        if time.time() > deadline:
            break
        if not _validate(base):
            continue
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 5:
            break
        result = _local_search(base, max_time=min(20.0, remaining_time))
        if len(result) > best_len:
            best_len = len(result)
            best_points = result
    
    if best_points is None or best_len < 5:
        best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
