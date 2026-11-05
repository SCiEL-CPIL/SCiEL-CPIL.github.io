import math
from itertools import combinations
from fractions import Fraction

def is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def dist_sq(p1, p2, D):
    # p1, p2 are (x, y) integers representing (x, y*sqrt(D))
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + D * dy * dy

def collinear(p1, p2, p3):
    # Determinant of the matrix with rows (x, y, 1)
    # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def concyclic(p1, p2, p3, p4, D):
    # Check if 4 points (x, y*sqrt(D)) are concyclic.
    # A circle in this metric: (x-cx)^2 + D(y-cy)^2 = r^2
    # x^2 + D y^2 - 2cx x - 2D cy y + (cx^2 + D cy^2 - r^2) = 0
    # Let A = -2cx, B = -2D cy, C = cx^2 + D cy^2 - r^2
    # x^2 + D y^2 + A x + B y + C = 0
    # We can set up a linear system for A, B, C using 3 points, and check the 4th.
    # x1^2 + D y1^2 + A x1 + B y1 + C = 0
    # x2^2 + D y2^2 + A x2 + B y2 + C = 0
    # x3^2 + D y3^2 + A x3 + B y3 + C = 0
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # System:
    # x1 A + y1 B + C = -(x1^2 + D y1^2)
    # x2 A + y2 B + C = -(x2^2 + D y2^2)
    # x3 A + y3 B + C = -(x3^2 + D y3^2)
    
    # Subtract eq1 from eq2 and eq3 to eliminate C:
    # (x2-x1) A + (y2-y1) B = -(x2^2 - x1^2 + D(y2^2 - y1^2))
    # (x3-x1) A + (y3-y1) B = -(x3^2 - x1^2 + D(y3^2 - y1^2))
    
    a11 = x2 - x1
    a12 = y2 - y1
    b1 = -(x2*x2 - x1*x1 + D * (y2*y2 - y1*y1))
    
    a21 = x3 - x1
    a22 = y3 - y1
    b2 = -(x3*x3 - x1*x1 + D * (y3*y3 - y1*y1))
    
    det = a11 * a22 - a12 * a21
    if det == 0:
        # Collinear, no unique circle (or degenerate)
        return False
        
    # Solve for A, B using Cramer's rule
    # A = (b1 * a22 - b2 * a12) / det
    # B = (a11 * b2 - a21 * b1) / det
    
    # We don't need A, B explicitly. We need to check if p4 satisfies the circle equation.
    # The circle equation is:
    # x^2 + D y^2 + A x + B y + C = 0
    # We can express A and B in terms of the determinant.
    # Let's compute the value of the circle equation at p4.
    # E4 = x4^2 + D y4^2 + A x4 + B y4 + C
    # We know C = -(x1^2 + D y1^2) - A x1 - B y1
    # E4 = x4^2 + D y4^2 + A(x4 - x1) + B(y4 - y1) - (x1^2 + D y1^2)
    # E4 = (x4^2 - x1^2 + D(y4^2 - y1^2)) + A(x4 - x1) + B(y4 - y1)
    
    # Substitute A and B:
    # A = (b1 a22 - b2 a12) / det
    # B = (a11 b2 - a21 b1) / det
    
    # E4 * det = (x4^2 - x1^2 + D(y4^2 - y1^2)) * det 
    #            + (x4 - x1) * (b1 a22 - b2 a12) 
    #            + (y4 - y1) * (a11 b2 - a21 b1)
            
    val = (x4*x4 - x1*x1 + D * (y4*y4 - y1*y1)) * det
    val += (x4 - x1) * (b1 * a22 - b2 * a12)
    val += (y4 - y1) * (a11 * b2 - a21 * b1)
    
    return val == 0

def construct():
    best_set = None
    best_D = None
    best_size = 0
    
    # Try small squarefree D
    Ds = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31]
    
    # Search range
    # For D=1, we can search larger. For other D, smaller might be enough if the set is small.
    # Let's start with a moderate range.
    R = 100
    
    for D in Ds:
        # Start with a base set of 2 points: (0,0) and (1,0)
        # Note: (1,0) has dist 1 from (0,0).
        base = [(0, 0), (1, 0)]
        
        # Backtracking search
        # We want to find a set of size > best_size
        # To save time, we can limit the search depth or use heuristics.
        # However, backtracking can explode.
        # Alternative: Randomized search with local improvement?
        # Let's try a simple iterative extension with a limited candidate pool.
        
        # Generate candidate points in range [-R, R] x [-R, R]
        # This is too many (200*200 = 40000).
        # We filter candidates by distance to base points.
        
        # Let's try a different strategy: 
        # Start with 3 points.
        # (0,0), (1,0), (x,y).
        # d(0,0)-(1,0) = 1.
        # d(0,0)-(x,y)^2 = x^2 + D y^2 = a^2
        # d(1,0)-(x,y)^2 = (x-1)^2 + D y^2 = b^2
        # x^2 + D y^2 = a^2
        # x^2 - 2x + 1 + D y^2 = b^2
        # a^2 - 2x + 1 = b^2 => 2x = a^2 - b^2 + 1 => x = (a^2 - b^2 + 1)/2
        # So x must be half-integer or integer.
        # If we restrict to integer x, then a^2 - b^2 + 1 must be even.
        
        # This algebraic constraint allows us to generate valid 3-point sets efficiently.
        
        max_points_found = 0
        best_set_for_D = None
        
        # Iterate over possible 'a' and 'b'
        # a, b are integers such that a^2 = x^2 + D y^2
        # Max distance squared approx (2R)^2 + D(2R)^2.
        # Let's limit a, b to 300.
        
        for a in range(1, 301):
            for b in range(1, 301):
                # 2x = a^2 - b^2 + 1
                num = a*a - b*b + 1
                if num % 2 != 0:
                    continue
                x = num // 2
                # x^2 + D y^2 = a^2
                # D y^2 = a^2 - x^2
                rhs = a*a - x*x
                if rhs < 0:
                    continue
                if rhs % D != 0:
                    continue
                y2 = rhs // D
                if not is_square(y2):
                    continue
                y = math.isqrt(y2)
                # We have a point (x, y). 
                # We should also consider (x, -y) and (-x, y) etc?
                # The equation was derived from distance to (0,0) and (1,0).
                # The point (x,y) is valid.
                # We should check if it's distinct from (0,0) and (1,0).
                if (x, y) in [(0,0), (1,0)]:
                    continue
                # Also check collinearity with (0,0) and (1,0)
                if collinear((0,0), (1,0), (x,y)):
                    continue
                
                S = [(0, 0), (1, 0), (x, y)]
                
                # Try to extend S
                # We can try to add points from a pre-generated list of "good" candidates
                # or search in a smaller box around the centroid.
                
                # For now, just record this 3-point set.
                if len(S) > max_points_found:
                    max_points_found = len(S)
                    best_set_for_D = S[:]
                
                # Extension search
                # To keep it fast, we only try to extend if we have time.
                # Let's try to add a 4th point.
                # The 4th point must have integer distances to (0,0), (1,0), (x,y).
                
                # This is getting complex. Let's just return the best 3-point set found?
                # No, we want more.
                
                # Let's use a simple heuristic: try to add points from a small set of candidates
                # that are "near" the existing points.
                
                # Actually, for the purpose of this improvement, finding a 6-point set is the goal.
                # Known 6-point sets for D=1:
                # (0,0), (1,0), (5,12), (16,0), (25,24), (41,0) ? No.
                # Let's rely on the search.
                
                # To avoid exponential blowup, we limit the extension search.
                # We'll try to add points from a pool of candidates that satisfy distance to first 2 points.
                
                # Generate candidates for 4th point
                # Must have dist to (0,0) = c, dist to (1,0) = d.
                # 2x = c^2 - d^2 + 1
                # D y^2 = c^2 - x^2
                
                # This is the same logic.
                # We can pre-compute all points (x,y) that are valid with (0,0) and (1,0).
                # Then check pairwise distances among them.
                
                # Let's do that.
        
        # Pre-compute all valid points P such that d(P, (0,0)) and d(P, (1,0)) are integers.
        valid_points = []
        for a in range(1, 301):
            for b in range(1, 301):
                num = a*a - b*b + 1
                if num % 2 != 0:
                    continue
                x = num // 2
                rhs = a*a - x*x
                if rhs < 0:
                    continue
                if rhs % D != 0:
                    continue
                y2 = rhs // D
                if not is_square(y2):
                    continue
                y = math.isqrt(y2)
                # Include both y and -y if y != 0
                if (x, y) not in [(0,0), (1,0)]:
                    valid_points.append((x, y))
                if y != 0 and (x, -y) not in [(0,0), (1,0)]:
                    valid_points.append((x, -y))
        
        # Remove duplicates
        valid_points = list(set(valid_points))
        
        # Now find the largest clique in the graph where edges exist if distance is integer,
        # and no 3 collinear, no 4 concyclic.
        # This is a Maximum Independent Set / Clique problem on a graph of size ~N.
        # N might be a few hundred.
        # We can use a greedy or backtracking approach to find a large subset.
        
        # Initialize with (0,0) and (1,0)
        S = [(0, 0), (1, 0)]
        
        # Add points one by one from valid_points
        # Sort valid_points by some heuristic?
        
        # To ensure we find a large set, we try to add points that are compatible with S.
        # We iterate through valid_points and try to add them.
        # If we can't add any, we might need to backtrack.
        
        # Simple Greedy:
        # Start with S = [(0,0), (1,0)]
        # For each p in valid_points:
        #   if compatible(p, S): S.append(p)
        
        # This greedy might get stuck.
        # Better: Randomized restarts.
        
        best_local = None
        for _ in range(100): # 100 restarts
            S = [(0, 0), (1, 0)]
            # Shuffle valid_points
            import random
            random.shuffle(valid_points)
            
            for p in valid_points:
                if p in S:
                    continue
                # Check distances
                ok = True
                for q in S:
                    d2 = dist_sq(p, q, D)
                    if not is_square(d2):
                        ok = False
                        break
                if not ok:
                    continue
                # Check collinearity
                for q1, q2 in combinations(S, 2):
                    if collinear(p, q1, q2):
                        ok = False
                        break
                if not ok:
                    continue
                # Check concyclicity
                for q1, q2, q3 in combinations(S, 3):
                    if concyclic(p, q1, q2, q3, D):
                        ok = False
                        break
                if ok:
                    S.append(p)
            
            if best_local is None or len(S) > len(best_local):
                best_local = S
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                    best_D = D
                    
        if best_local:
            if len(best_local) > best_size:
                best_size = len(best_local)
                best_set = best_local[:]
                best_D = D

    if best_set is None:
        # Fallback to a known small set
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[5, 1], [12, 1]]]}
        
    return {'D': best_D, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
