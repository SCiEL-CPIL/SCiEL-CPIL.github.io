import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def dist_sq(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def check_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def check_concyclic(p1, p2, p3, p4):
        # Determinant of the matrix for circle equation
        # x^2 + y^2 + Dx + Ey + F = 0
        # | x1^2+y1^2 x1 y1 1 |
        # | x2^2+y2^2 x2 y2 1 | = 0
        # | x3^2+y3^2 x3 y3 1 |
        # | x4^2+y4^2 x4 y4 1 |
        
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        d1 = x1*x1 + y1*y1
        d2 = x2*x2 + y2*y2
        d3 = x3*x3 + y3*y3
        d4 = x4*x4 + y4*y4
        
        # Expand determinant
        # Using Sarrus rule or expansion
        det = (d1 * (x2*(y3*1 - 1*y4) - y2*(x3*1 - 1*x4) + 1*(x3*y4 - y3*x4)) -
               x1 * (d2*(y3*1 - 1*y4) - y2*(d3*1 - 1*d4) + 1*(d3*y4 - y3*d4)) +
               y1 * (d2*(x3*1 - 1*x4) - d3*(x2*1 - 1*x4) + 1*(x2*d4 - d3*x4)) -
               1 * (d2*(x3*y4 - y3*x4) - y2*(d3*y4 - y3*d4) + d3*(x2*y4 - y2*x4)))
        return det == 0

    def is_valid_set(points):
        n = len(points)
        if n < 3:
            return False
        # Check distinct
        for i in range(n):
            for j in range(i+1, n):
                if points[i] == points[j]:
                    return False
        # Check distances
        for i in range(n):
            for j in range(i+1, n):
                if not is_square(dist_sq(points[i], points[j])):
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

    # Initial 5-point set from seed
    initial_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_points = initial_points
    
    # Generate candidate points
    # We search for a 6th point that can be added to the initial 5.
    # If that fails, we might try other 5-point bases, but let's stick to this family first.
    
    # To allow for a 6th point, we need a point (x,y) such that:
    # 1. dist_sq to all 5 is a square.
    # 2. Not collinear with any pair of the 5.
    # 3. Not concyclic with any triple of the 5.
    
    # Let's search for such a point in a larger grid.
    # The initial points are roughly in [-56, 16] x [0, 63].
    # Let's search in [-100, 100] x [-100, 100].
    
    found_6 = False
    best_6_points = None
    
    # Precompute squares for distance check up to max possible dist^2
    # Max dist in 200x200 box is ~280. Dist^2 ~ 80000.
    # But we can just use is_square.
    
    # Optimization: 
    # For a new point P, check against existing set S.
    # If S is valid, P must satisfy:
    # - dist_sq(P, p) is square for all p in S
    # - no 3 collinear
    # - no 4 concyclic
    
    # We iterate x, y
    # To speed up, we can pre-filter candidates that have integer distance to (0,0)
    # i.e., x^2 + y^2 is a square.
    
    # Let's generate all integer points (x,y) in range [-100, 100]
    # such that x^2+y^2 is a square.
    # Then check against the other 4 points.
    
    candidates = []
    limit = 100
    for x in range(-limit, limit+1):
        for y in range(-limit, limit+1):
            if is_square(x*x + y*y):
                candidates.append((x, y))
                
    # Sort candidates by distance to centroid or something? 
    # Just iterate.
    
    # We want to find P in candidates such that {initial_points} U {P} is valid.
    # We already know initial_points is valid.
    
    for p in candidates:
        if p in initial_points:
            continue
            
        # Check distances to all 5
        ok_dist = True
        for q in initial_points:
            if not is_square(dist_sq(p, q)):
                ok_dist = False
                break
        if not ok_dist:
            continue
            
        # Check collinearity
        ok_coll = True
        for i in range(5):
            for j in range(i+1, 5):
                if check_collinear(p, initial_points[i], initial_points[j]):
                    ok_coll = False
                    break
            if not ok_coll:
                break
        if not ok_coll:
            continue
            
        # Check concyclicity
        ok_conc = True
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    if check_concyclic(p, initial_points[i], initial_points[j], initial_points[k]):
                        ok_conc = False
                        break
                if not ok_conc:
                    break
            if not ok_conc:
                break
        if not ok_conc:
            continue
            
        # Found a valid 6th point
        found_6 = True
        best_6_points = initial_points + [p]
        break
        
    if found_6:
        points = best_6_points
    else:
        points = initial_points
        # If we have time, we could try to find a better 5-point set or a 6-point set from a different base.
        # But 5 is the baseline. 6 is the target.
        
    # Convert to required format
    # D=1
    # points: [[[x_num, x_den], [y_num, y_den]], ...]
    result_points = []
    for x, y in points:
        result_points.append([[x, 1], [y, 1]])
        
    return {'D': 1, 'points': result_points}
