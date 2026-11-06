"""
Search for integral point sets in general position (no 3 collinear, no 4 concyclic)
in the plane with points (x, y*sqrt(D)) for rational x, y.
"""
import time
import random
from math import gcd, isqrt
from fractions import Fraction

DEADLINE = 145.0  # safety margin before 150s kill

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _sq_dist_int(p, q, D):
    """Squared distance between integer points p=(x1,y1), q=(x2,y2) in (x, y*sqrt(D)) space.
    Returns integer."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + D * dy * dy

def _dist_rational_int(p, q, D):
    """Check if squared distance is a perfect square (of an integer) for integer points."""
    d2 = _sq_dist_int(p, q, D)
    return _is_square(d2)

def _no_three_collinear_int(points):
    """Check no three integer points are collinear using cross product = 0."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def _det4_int(m):
    """Compute determinant of 4x4 integer matrix using exact arithmetic."""
    a, b, c, d = m[0], m[1], m[2], m[3]
    return (a[0]*(b[1]*c[2]*d[3] - b[1]*d[2]*c[3] - b[2]*c[1]*d[3] + b[2]*d[1]*c[3] + b[3]*c[1]*d[2] - b[3]*d[1]*c[2])
          - a[1]*(b[0]*c[2]*d[3] - b[0]*d[2]*c[3] - b[2]*c[0]*d[3] + b[2]*d[0]*c[3] + b[3]*c[0]*d[2] - b[3]*d[0]*c[2])
          + a[2]*(b[0]*c[1]*d[3] - b[0]*d[1]*c[3] - b[1]*c[0]*d[3] + b[1]*d[0]*c[3] + b[3]*c[0]*d[1] - b[3]*d[0]*c[1])
          - a[3]*(b[0]*c[1]*d[2] - b[0]*d[1]*c[2] - b[1]*c[0]*d[2] + b[1]*d[0]*c[2] + b[2]*c[0]*d[1] - b[2]*d[0]*c[1]))

def _no_four_concyclic_int(points, D):
    """Check no four integer points are concyclic for given D."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    M = [
                        [p1[0]*p1[0] + D*p1[1]*p1[1], p1[0], p1[1], 1],
                        [p2[0]*p2[0] + D*p2[1]*p2[1], p2[0], p2[1], 1],
                        [p3[0]*p3[0] + D*p3[1]*p3[1], p3[0], p3[1], 1],
                        [p4[0]*p4[0] + D*p4[1]*p4[1], p4[0], p4[1], 1],
                    ]
                    if _det4_int(M) == 0:
                        return False
    return True

def _all_distances_rational_int(points, D):
    """Check all pairwise distances are rational (integer squared distances are perfect squares)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_rational_int(points[i], points[j], D):
                return False
    return True

def _is_valid_int(points, D):
    """Check validity for integer points."""
    if len(points) < 3:
        return False
    if not _all_distances_rational_int(points, D):
        return False
    if not _no_three_collinear_int(points):
        return False
    if not _no_four_concyclic_int(points, D):
        return False
    return True

def _try_add_point_int(current, new_point, D):
    """Try adding a point to current set, checking only new constraints."""
    # Check distance to all existing points
    for p in current:
        if not _dist_rational_int(new_point, p, D):
            return False
    # Check collinearity with all pairs
    n = len(current)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = current[i]
            x2, y2 = current[j]
            x3, y3 = new_point
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross == 0:
                return False
    # Check concyclicity with all triples
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = current[i], current[j], current[k]
                M = [
                    [p1[0]*p1[0] + D*p1[1]*p1[1], p1[0], p1[1], 1],
                    [p2[0]*p2[0] + D*p2[1]*p2[1], p2[0], p2[1], 1],
                    [p3[0]*p3[0] + D*p3[1]*p3[1], p3[0], p3[1], 1],
                    [new_point[0]*new_point[0] + D*new_point[1]*new_point[1], new_point[0], new_point[1], 1],
                ]
                if _det4_int(M) == 0:
                    return False
    return True

def _search_incremental_int(D, start_points, max_coord, deadline):
    """Incremental search: try to add points one at a time."""
    current = list(start_points)
    best = current[:]
    best_len = len(current)
    
    # Try adding points in a systematic manner
    for attempt in range(1000000):
        if time.time() > deadline:
            break
        if len(current) >= 12:
            break
        
        # Generate candidate point
        if attempt < 50000:
            # Systematic grid search
            x = (attempt % (2*max_coord+1)) - max_coord
            y = ((attempt // (2*max_coord+1)) % (2*max_coord+1)) - max_coord
        else:
            # Random search
            x = random.randint(-max_coord, max_coord)
            y = random.randint(-max_coord, max_coord)
        
        new_point = (x, y)
        
        # Check if point is already in set
        if new_point in current:
            continue
        
        if _try_add_point_int(current, new_point, D):
            current.append(new_point)
            best = current[:]
            best_len = len(current)
            if best_len >= 8:
                break
    
    return best

def construct():
    start_time = time.time()
    
    # Start with the known 5-point integral set
    known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = known_points[:]
    best_D = 1
    best_len = 5
    
    # Verify the known set with fast integer checks
    if _is_valid_int(known_points, 1):
        pass  # confirmed valid
    
    deadline = start_time + DEADLINE
    
    # Phase 1: Try to extend the known 5-point set with D=1
    current = list(known_points)
    phase1_deadline = min(deadline - 30, start_time + 60)
    extended = _search_incremental_int(1, current, 100, phase1_deadline)
    if len(extended) > best_len:
        best_points = extended[:]
        best_len = len(extended)
    
    # Phase 2: Try different starting configurations
    if time.time() < deadline - 20:
        # Try other known configurations or small sets
        alt_configs = [
            [(0, 0), (3, 0), (0, 4)],  # 3-4-5 triangle
            [(0, 0), (5, 0), (0, 12)],  # 5-12-13 triangle
            [(0, 0), (7, 0), (0, 24)],  # 7-24-25 triangle
        ]
        
        for config in alt_configs:
            if time.time() > deadline - 10:
                break
            if not _is_valid_int(config, 1):
                continue
            sub_deadline = min(deadline - 5, time.time() + 15)
            extended = _search_incremental_int(1, config, 80, sub_deadline)
            if len(extended) > best_len:
                best_points = extended[:]
                best_len = len(extended)
    
    # Phase 3: Try D=2 with smaller search
    if time.time() < deadline - 10:
        for D_val in [2, 3]:
            if time.time() > deadline - 5:
                break
            # Start with small valid sets for this D
            # For D=2, try simple configurations
            test_configs = [
                [(0, 0), (1, 0)],  # distance = 1
                [(0, 0), (1, 1)],  # distance = sqrt(1+2) = sqrt(3) - not rational
            ]
            
            # Find a valid starting set
            start_set = None
            for cfg in test_configs:
                if _is_valid_int(cfg, D_val):
                    start_set = cfg
                    break
            
            if start_set is None:
                # Try to find any valid 3-point set
                for _ in range(5000):
                    if time.time() > deadline - 5:
                        break
                    p1 = (random.randint(-10, 10), random.randint(-10, 10))
                    p2 = (random.randint(-10, 10), random.randint(-10, 10))
                    p3 = (random.randint(-10, 10), random.randint(-10, 10))
                    if len({p1, p2, p3}) == 3 and _is_valid_int([p1, p2, p3], D_val):
                        start_set = [p1, p2, p3]
                        break
            
            if start_set:
                sub_deadline = min(deadline - 2, time.time() + 10)
                extended = _search_incremental_int(D_val, start_set, 30, sub_deadline)
                if len(extended) > best_len:
                    best_points = extended[:]
                    best_D = D_val
                    best_len = len(extended)
    
    # Format result
    result_points = []
    for (x, y) in best_points:
        result_points.append([[x, 1], [y, 1]])
    
    return {'D': best_D, 'points': result_points}
