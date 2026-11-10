import random
import math
import time

# EVOLVE-BLOCK-START
def construct():
    # Target: Maximize points in general position (no 3 collinear, no 4 concyclic)
    # with rational distances.
    # We use D=1 (integer coordinates) for simplicity, as integer points with 
    # integer pairwise distances form a subset of the valid configurations.
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def are_collinear(p1, p2, p3):
        # Determinant check
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def are_concyclic(p1, p2, p3, p4):
        # Check if 4 points lie on a circle.
        # We can check if the circumcircle of p1, p2, p3 contains p4.
        # Or use the determinant of the matrix:
        # | x^2+y^2  x  y  1 |
        # All 4 rows must be linearly dependent.
        
        def det4(m):
            # Calculate determinant of 4x4 matrix
            # Using cofactor expansion or simple permutation sum (16 terms)
            # For efficiency and simplicity, we use the formula for circle equation:
            # A x^2 + A y^2 + B x + C y + D = 0
            # Solve for A, B, C, D for first 3 points, check 4th.
            
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            
            # Equations:
            # x1^2+y1^2 + B x1 + C y1 + D = 0
            # x2^2+y2^2 + B x2 + C y2 + D = 0
            # x3^2+y3^2 + B x3 + C y3 + D = 0
            
            # Subtract eq1 from eq2 and eq3 to eliminate D
            # B(x2-x1) + C(y2-y1) = -(x2^2+y2^2 - x1^2-y1^2)
            # B(x3-x1) + C(y3-y1) = -(x3^2+y3^2 - x1^2-y1^2)
            
            a11 = x2 - x1
            a12 = y2 - y1
            b1 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
            
            a21 = x3 - x1
            a22 = y3 - y1
            b2 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
            
            det = a11 * a22 - a12 * a21
            if det == 0:
                return True # Collinear, technically concyclic with radius infinity? 
                            # But task says no 3 collinear, so this case shouldn't arise 
                            # if we check collinearity first. If it does, treat as concyclic 
                            # to be safe? No, if 3 are collinear, they don't define a unique circle.
                            # However, the constraint is "no 4 concyclic". If 3 are collinear, 
                            # they are not on a finite circle. So this check is only valid 
                            # for non-collinear triples. We assume non-collinearity.
                return False
            
            # Cramer's rule for B and C
            B = (b1 * a22 - a12 * b2) / det
            C = (a11 * b2 - b1 * a21) / det
            
            # D = -(x1^2+y1^2 + B*x1 + C*y1)
            # Check p4:
            x4, y4 = p4
            val = x4*x4 + y4*y4 + B*x4 + C*y4 + (-(x1*x1 + y1*y1 + B*x1 + C*y1))
            
            # Use exact rational arithmetic to avoid floating point errors
            # val = 0 ?
            # B and C are rationals. Let's do it with fractions to be exact.
            
            from fractions import Fraction
            B_frac = Fraction(b1 * a22 - a12 * b2, det)
            C_frac = Fraction(a11 * b2 - b1 * a21, det)
            D_frac = Fraction(-(x1*x1 + y1*y1) - B_frac*x1 - C_frac*y1, 1)
            
            check = Fraction(x4*x4 + y4*y4) + B_frac*x4 + C_frac*y4 + D_frac
            return check == 0

    def check_set(points):
        n = len(points)
        # Check pairwise distances are rational (integers here)
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                if not is_square(dx*dx + dy*dy):
                    return False
        
        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Initial known valid set of 5 points
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    start_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_set = start_points[:]
    
    # Generate candidate points
    # We want points that are at integer distance from existing points.
    # A good pool is points on a grid, but we must filter.
    # To find more points, we can search in a larger area.
    
    # Pre-generate a pool of candidates
    # We'll use a grid from -200 to 200.
    candidates = []
    for x in range(-200, 201):
        for y in range(-200, 201):
            if (x, y) not in start_points:
                candidates.append((x, y))
    
    # Shuffle candidates
    random.seed(42) # For reproducibility, but we can also randomize per run
    random.shuffle(candidates)
    
    # Greedy addition
    current_set = start_points[:]
    
    # We will try to add points one by one.
    # To make it faster, we can pre-filter candidates that have integer distance to all points in current_set.
    # But current_set changes.
    
    # Let's do a few iterations of greedy addition with random restarts or local search.
    # Given time limit, a single deep greedy search with a large pool is good.
    
    # Optimize: Instead of checking all candidates against all points in current_set repeatedly,
    # we can just iterate through the shuffled list and try to add.
    
    for p in candidates:
        # Check if p has integer distance to all points in current_set
        valid_dist = True
        for q in current_set:
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            if not is_square(dx*dx + dy*dy):
                valid_dist = False
                break
        
        if not valid_dist:
            continue
            
        # Check general position with current_set + p
        new_set = current_set + [p]
        if check_set(new_set):
            current_set = new_set
            if len(current_set) > len(best_set):
                best_set = current_set[:]
                # If we reach 8, we can stop or continue? 
                # Task says maximize. 8 is "new". Let's aim for max possible.
                # But 120s limit.
                # If we find 8, it's a great score.
                # Let's continue to see if we can get 9, but it's unlikely.
    
    # If we didn't find much, maybe try a different starting set or search space?
    # For now, return the best found.
    
    # Format output
    # D=1
    # points: [[[x_num, x_den], [y_num, y_den]], ...]
    points_out = []
    for x, y in best_set:
        points_out.append([[x, 1], [y, 1]])
        
    return {'D': 1, 'points': points_out}

# EVOLVE-BLOCK-END
