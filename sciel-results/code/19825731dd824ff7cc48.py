import math
from itertools import combinations

def is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def dist_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx*dx + dy*dy

def collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # d = 2 * (x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2))
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if d == 0:
        # p1, p2, p3 are collinear. If p4 is also on the line, they are "concyclic" in a degenerate sense?
        # But the problem states no three collinear in the final set.
        # If p1,p2,p3 are collinear, they don't define a unique circle.
        # In the context of adding p4 to a set with no three collinear, this case shouldn't arise for p1,p2,p3 from S.
        # However, if we are checking if p4 makes a concyclic set with p1,p2,p3, and p1,p2,p3 are collinear, 
        # then p1,p2,p3,p4 are concyclic only if p4 is also on that line (degenerate circle) or if we consider the limit.
        # Standard definition: 4 points are concyclic if they lie on a circle. 3 collinear points do not lie on a circle.
        # So if p1,p2,p3 are collinear, they cannot be part of a non-degenerate circle.
        # Thus, if p1,p2,p3 are collinear, the set {p1,p2,p3,p4} is not concyclic (in the standard sense of lying on a circle).
        # Wait, the condition "no four concyclic" means there is no circle passing through all 4.
        # If p1,p2,p3 are collinear, no circle passes through them. So no circle passes through all 4.
        # So they are NOT concyclic.
        return False

    # U = x1^2+y1^2(y2-y3) + ...
    # V = x1^2+y1^2(x3-x2) + ...
    c1 = x1*x1 + y1*y1
    c2 = x2*x2 + y2*y2
    c3 = x3*x3 + y3*y3
    
    U = c1 * (y2 - y3) + c2 * (y3 - y1) + c3 * (y1 - y2)
    V = c1 * (x3 - x2) + c2 * (x1 - x3) + c3 * (x2 - x1)
    
    # Check if (x4, y4) satisfies (x4 - U/d)^2 + (y4 - V/d)^2 = r^2
    # Equivalent to (d*x4 - U)^2 + (d*y4 - V)^2 = (d*x1 - U)^2 + (d*y1 - V)^2
    lhs = (d*x4 - U)**2 + (d*y4 - V)**2
    rhs = (d*x1 - U)**2 + (d*y1 - V)**2
    
    return lhs == rhs

def construct():
    # Initial set of 5 points (known rational distance set)
    S = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # We will try to expand the set by searching in a box.
    # Start with a moderate box and expand if stuck.
    # However, to ensure we finish in time, we'll do a few passes with increasing box sizes.
    
    box_sizes = [200, 300, 400]
    
    for box in box_sizes:
        found_new = True
        while found_new:
            found_new = False
            # Iterate through potential candidates
            # We scan y from -box to box, x from -box to box
            # To break ties or find "better" points, we could sort candidates, but that's expensive.
            # We just add the first valid one found.
            
            # Optimization: Pre-compute properties of S?
            # For each new point P, we check against all Q in S.
            
            for y in range(-box, box + 1):
                for x in range(-box, box + 1):
                    P = (x, y)
                    if P in S:
                        continue
                    
                    valid = True
                    # Check distances
                    for Q in S:
                        d_sq = dist_sq(P, Q)
                        if not is_square(d_sq):
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Check collinearity
                    for Q1, Q2 in combinations(S, 2):
                        if collinear(P, Q1, Q2):
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Check concyclicity
                    for Q1, Q2, Q3 in combinations(S, 3):
                        if concyclic(P, Q1, Q2, Q3):
                            valid = False
                            break
                    
                    if valid:
                        S.append(P)
                        found_new = True
                        break
                if found_new:
                    break
        
        # If no new point found in this box size, try the next larger box
        if not found_new:
            continue
            
    # If we still have few points, we might try a different seed or randomized search, 
    # but given the constraints, this deterministic expansion is the safest bet.
    
    # Convert to required format
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in S]}
