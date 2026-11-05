import math
from itertools import combinations

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def concyclic(p1, p2, p3, p4):
    # Determinant of the circle equation matrix
    # x^2 + y^2 + Dx + Ey + F = 0
    # x1^2+y1^2 x1 y1 1
    # x2^2+y2^2 x2 y2 1
    # x3^2+y3^2 x3 y3 1
    # x4^2+y4^2 x4 y4 1
    a1 = p1[0]**2 + p1[1]**2
    a2 = p2[0]**2 + p2[1]**2
    a3 = p3[0]**2 + p3[1]**2
    a4 = p4[0]**2 + p4[1]**2
    
    # Calculate determinant using expansion or Sarrus
    # For efficiency, we can check if p4 lies on the circle defined by p1, p2, p3
    # Circle through p1, p2, p3:
    # (x - cx)^2 + (y - cy)^2 = r^2
    # 2(x2-x1)x + 2(y2-y1)y = x2^2-x1^2 + y2^2-y1^2
    # 2(x3-x1)x + 2(y3-y1)y = x3^2-x1^2 + y3^2-y1^2
    # Solve for cx, cy
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if d == 0:
        # Points are collinear, so they don't define a unique circle, 
        # but the problem states no three collinear in the set.
        # If p1,p2,p3 are collinear, this case shouldn't happen for valid sets.
        return False 
        
    ux = ((x1*x1 + y1*y1) * (y2 - y3) + (x2*x2 + y2*y2) * (y3 - y1) + (x3*x3 + y3*y3) * (y1 - y2)) / d
    uy = ((x1*x1 + y1*y1) * (x3 - x2) + (x2*x2 + y2*y2) * (x1 - x3) + (x3*x3 + y3*y3) * (x2 - x1)) / d
    
    # Check if p4 is on the circle
    # (x4 - ux)^2 + (y4 - uy)^2 == (x1 - ux)^2 + (y1 - uy)^2
    # Multiply by d^2 to avoid floats?
    # Let's use exact rational arithmetic with fractions or integers.
    # ux = U/d, uy = V/d
    # (x4 - U/d)^2 + (y4 - V/d)^2 = (x1 - U/d)^2 + (y1 - U/d)^2
    # (d*x4 - U)^2 + (d*y4 - V)^2 = (d*x1 - U)^2 + (d*y1 - V)^2
    
    U = (x1*x1 + y1*y1) * (y2 - y3) + (x2*x2 + y2*y2) * (y3 - y1) + (x3*x3 + y3*y3) * (y1 - y2)
    V = (x1*x1 + y1*y1) * (x3 - x2) + (x2*x2 + y2*y2) * (x1 - x3) + (x3*x3 + y3*y3) * (x2 - x1)
    
    lhs = (d*x4 - U)**2 + (d*y4 - V)**2
    rhs = (d*x1 - U)**2 + (d*y1 - V)**2
    
    return lhs == rhs

def construct():
    # Initial set of 5 points
    S = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Search for more points
    # We try to add points one by one.
    # To keep it fast, we limit the search space and iterations.
    
    max_x = 200
    min_x = -200
    max_y = 200
    min_y = 0
    
    found_new = True
    while found_new:
        found_new = False
        # Iterate through potential candidates
        # We can optimize by checking distances first
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
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
                    # Break to restart search with new S? 
                    # Or continue? Restarting is safer to ensure we don't miss a better point for the next step,
                    # but adding the first found one is fine for maximization if we loop.
                    # Let's break to add this point and restart the outer loop.
                    break
            if found_new:
                break
        
        # If no new point found in this range, we might expand range or stop.
        # Given time constraints, we stop if no new point found in the current box.
        # We could expand the box slightly, but 200x200 is likely enough for small sets.
        
    # Convert to required format
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in S]}
