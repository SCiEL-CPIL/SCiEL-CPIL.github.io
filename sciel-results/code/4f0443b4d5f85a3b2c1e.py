import time
import math
from fractions import Fraction

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def check_rational_distance(p1, p2, D):
    # p1, p2 are (x_num, x_den, y_num, y_den) or just integers for D=1 search
    # For D=1, we search in integers
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dist_sq = dx * dx + D * dy * dy
    return is_square(dist_sq)

def check_collinear(p1, p2, p3):
    # p1, p2, p3 are (x, y)
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

def check_concyclic(p1, p2, p3, p4):
    # p1..p4 are (x, y)
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Determinant of the matrix:
    # x^2+y^2  x  y  1
    # ...
    # If det == 0, they are concyclic (or collinear, but we check collinear separately)
    
    # Using the formula for concyclicity
    # | x1^2+y1^2 x1 y1 1 |
    # | x2^2+y2^2 x2 y2 1 | = 0
    # | x3^2+y3^2 x3 y3 1 |
    # | x4^2+y4^2 x4 y4 1 |
    
    # To avoid large numbers, we can use the property that 4 points are concyclic
    # iff the angle subtended by a chord is equal. But determinant is safer.
    
    # Let's compute the determinant.
    # We can subtract row 4 from rows 1, 2, 3 to simplify.
    # Let r_i = (x_i^2+y_i^2, x_i, y_i) - (x_4^2+y_4^2, x_4, y_4)
    # Then det of the 3x3 matrix of these differences is 0.
    
    x4sq = x4*x4 + y4*y4
    r1 = (x1*x1 + y1*y1 - x4sq, x1 - x4, y1 - y4)
    r2 = (x2*x2 + y2*y2 - x4sq, x2 - x4, y2 - y4)
    r3 = (x3*x3 + y3*y3 - x4sq, x3 - x4, y3 - y4)
    
    # Det of:
    # r1[0] r1[1] r1[2]
    # r2[0] r2[1] r2[2]
    # r3[0] r3[1] r3[2]
    
    det = (r1[0] * (r2[1]*r3[2] - r2[2]*r3[1]) - 
           r1[1] * (r2[0]*r3[2] - r2[2]*r3[0]) + 
           r1[2] * (r2[0]*r3[1] - r2[1]*r3[0]))
    return det == 0

def is_valid_set(points, D):
    n = len(points)
    # Check pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            if not check_rational_distance(points[i], points[j], D):
                return False
    # Check collinearity
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if check_collinear(points[i], points[j], points[k]):
                    return False
    # Check concyclicity
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if check_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Start with the known 5-point set
    # (0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    # Verify base points
    if not is_valid_set(base_points, D):
        # Fallback to simpler set if base is invalid (should not happen)
        base_points = [(0,0), (3,4), (0,4)] # Triangle
        D = 1

    current_points = base_points[:]
    
    # Try to add more points
    # Search space: small integers
    # We can try to add points from a grid
    max_coord = 100
    
    # To speed up, we can pre-filter candidates that have rational distance to ALL existing points
    # But first, let's just try adding one point at a time.
    
    # We'll iterate through a grid of candidate points
    # We prioritize points that are "close" or have small coordinates
    
    # Generate candidates
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((x, y))
    
    # Sort candidates by some heuristic? Maybe distance to centroid?
    # Or just iterate.
    
    # We want to maximize the number of points.
    # Greedy approach: Add the first valid point found, then repeat.
    # This might get stuck in a local optimum.
    # Better: Try to add points until no more can be added.
    
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        # Remove candidates that are already in the set
        current_set = set(current_points)
        
        # Filter candidates that have rational distance to all current points
        valid_candidates = []
        for p in candidates:
            if p in current_set:
                continue
            
            # Check distance to all existing points
            ok = True
            for q in current_points:
                if not check_rational_distance(p, q, D):
                    ok = False
                    break
            if not ok:
                continue
            
            # Check collinearity with all pairs of existing points
            n = len(current_points)
            for i in range(n):
                for j in range(i+1, n):
                    if check_collinear(p, current_points[i], current_points[j]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            # Check concyclicity with all triples of existing points
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if check_concyclic(p, current_points[i], current_points[j], current_points[k]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            valid_candidates.append(p)
        
        if valid_candidates:
            # Pick the first one (or the one with smallest norm to keep numbers small)
            # Sorting by norm might be good to keep integers small
            valid_candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
            new_point = valid_candidates[0]
            current_points.append(new_point)
            added_any = True
            # Remove it from candidates list to avoid re-checking
            candidates.remove(new_point)
            
            # Check deadline
            if time.time() > deadline - 5.0:
                break
    
    # Convert to required format
    # points: [[[x_num, x_den], [y_num, y_den]], ...]
    # D is 1
    
    final_points = []
    for x, y in current_points:
        final_points.append([[x, 1], [y, 1]])
        
    return {'D': D, 'points': final_points}
