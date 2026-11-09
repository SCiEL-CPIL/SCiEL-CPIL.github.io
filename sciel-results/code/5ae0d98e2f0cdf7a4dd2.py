import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _are_concyclic(p1, p2, p3, p4):
    a1, b1 = p1
    a2, b2 = p2
    a3, b3 = p3
    a4, b4 = p4
    
    if _are_collinear(p1, p2, p3):
        return False
    
    rhs1 = -(a2**2 - a1**2 + b2**2 - b1**2)
    rhs2 = -(a3**2 - a1**2 + b3**2 - b1**2)
    
    x1 = a2 - a1
    y1 = b2 - b1
    x2 = a3 - a1
    y2 = b3 - b1
    
    det = x1*y2 - x2*y1
    if det == 0:
        return False
    
    num_D = rhs1*y2 - rhs2*y1
    num_E = x1*rhs2 - x2*rhs1
    num_F = -((a1**2 + b1**2)*det + num_D*a1 + num_E*b1)
    
    check = (a4**2 + b4**2)*det + num_D*a4 + num_E*b4 + num_F
    return check == 0

def _validate_point_set(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx*dx + dy*dy
            if not _is_square(d2):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _try_add_point(base_points, x, y):
    new_point = (x, y)
    if new_point in base_points:
        return False
    for p in base_points:
        dx = x - p[0]
        dy = y - p[1]
        d2 = dx*dx + dy*dy
        if not _is_square(d2):
            return False
    n = len(base_points)
    for i in range(n):
        for j in range(i+1, n):
            if _are_collinear(base_points[i], base_points[j], new_point):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_concyclic(base_points[i], base_points[j], base_points[k], new_point):
                    return False
    return True

def _find_candidates_with_int_dist(base_points, max_coord):
    candidates = set()
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            valid = True
            for p in base_points:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx*dx + dy*dy
                if not _is_square(d2):
                    valid = False
                    break
            if valid:
                candidates.add((x, y))
    return list(candidates)

def _try_known_6point_sets():
    """Try known or constructed 6-point rational distance sets."""
    # Try various configurations
    test_sets = [
        # Hexagon-like: (0,0), (a,0), (a/2, a*sqrt(3)/2) won't work with D=1
        # Try a different approach: use points from known rational distance sets
        [(0, 0), (3, 4), (5, 12), (0, 5), (16, 63), (-16, 63)],
        [(0, 0), (3, 4), (5, 12), (0, 5), (16, 63), (16, -63)],
        [(0, 0), (3, 4), (5, 12), (0, 5), (-16, 63), (-16, -63)],
        [(0, 0), (3, 4), (5, 12), (0, 5), (16, 63), (0, 63)],
        [(0, 0), (3, 4), (5, 12), (0, 5), (16, 63), (-56, 33)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (0, 63)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (56, 33)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (-56, -33)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (56, -33)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (32, 0)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (-32, 0)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (0, -33)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (0, 96)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (48, 0)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (-48, 0)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (0, 126)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (112, 66)],
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63), (-112, 66)],
    ]
    
    for pts in test_sets:
        if len(pts) != 6:
            continue
        # Check all distinct
        if len(set(pts)) != 6:
            continue
        if _validate_point_set(pts):
            return pts
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # First try known 6-point sets
    known6 = _try_known_6point_sets()
    if known6 is not None:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known6]}
    
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _validate_point_set(base5):
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[0, 1], [5, 1]]]}
    
    best_set = base5.copy()
    
    # Search for 6th point with increasing radius
    for max_coord in [50, 80, 100, 150, 200, 300, 400, 500]:
        if time.time() > deadline - 5:
            break
        
        candidates = _find_candidates_with_int_dist(base5, max_coord)
        candidates = [p for p in candidates if p not in base5]
        
        cx = sum(p[0] for p in base5) / 5
        cy = sum(p[1] for p in base5) / 5
        candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        
        for cand in candidates:
            if time.time() > deadline - 2:
                break
            if _try_add_point(base5, cand[0], cand[1]):
                new_set = base5 + [cand]
                if _validate_point_set(new_set):
                    best_set = new_set
                    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
    
    # If we found a 6th point, try to find a 7th
    if len(best_set) == 6 and time.time() < deadline - 10:
        for max_coord in [50, 80, 100]:
            if time.time() > deadline - 3:
                break
            candidates7 = _find_candidates_with_int_dist(best_set, max_coord)
            candidates7 = [p for p in candidates7 if p not in best_set]
            cx = sum(p[0] for p in best_set) / len(best_set)
            cy = sum(p[1] for p in best_set) / len(best_set)
            candidates7.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
            
            for cand in candidates7:
                if time.time() > deadline - 1:
                    break
                if _try_add_point(best_set, cand[0], cand[1]):
                    new_set = best_set + [cand]
                    if _validate_point_set(new_set):
                        best_set = new_set
                        break
            if len(best_set) == 7:
                break
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

# EVOLVE-BLOCK-END
