import math
from itertools import combinations

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def dist_sq(p1, p2, D):
    return (p1[0] - p2[0])**2 + D * (p1[1] - p2[1])**2

def collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def concyclic(p1, p2, p3, p4, D):
    # Circle in metric: (x-cx)^2 + D(y-cy)^2 = r^2
    # x^2 - 2cx x + cx^2 + D y^2 - 2D cy y + D cy^2 = r^2
    # x^2 + D y^2 - 2cx x - 2D cy y + (cx^2 + D cy^2 - r^2) = 0
    # A x + B y + C = 0 where A=-2cx, B=-2D cy, C=...
    # Actually, 3 points define a unique circle (ellipse).
    # We can check if p4 satisfies the equation derived from p1,p2,p3.
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Equation: x^2 + D y^2 + U x + V y + W = 0
    # x1^2 + D y1^2 + U x1 + V y1 + W = 0
    # x2^2 + D y2^2 + U x2 + V y2 + W = 0
    # x3^2 + D y3^2 + U x3 + V y3 + W = 0
    
    # Subtract 1 from 2 and 3 to eliminate W
    # U(x2-x1) + V(y2-y1) = -(x2^2-x1^2 + D(y2^2-y1^2))
    # U(x3-x1) + V(y3-y1) = -(x3^2-x1^2 + D(y3^2-y1^2))
    
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x3 - x1
    dy2 = y3 - y1
    
    rhs1 = -(x2*x2 - x1*x1 + D*(y2*y2 - y1*y1))
    rhs2 = -(x3*x3 - x1*x1 + D*(y3*y3 - y1*y1))
    
    det = dx1 * dy2 - dx2 * dy1
    if det == 0:
        return False # Collinear, handled elsewhere or invalid circle
        
    U = (rhs1 * dy2 - rhs2 * dy1) / det
    V = (dx1 * rhs2 - dx2 * rhs1) / det
    
    # Check p4
    # x4^2 + D y4^2 + U x4 + V y4 + W = 0
    # W = -(x1^2 + D y1^2 + U x1 + V y1)
    val = x4*x4 + D*y4*y4 + U*x4 + V*y4 - (x1*x1 + D*y1*y1 + U*x1 + V*y1)
    return val == 0

def construct():
    best_points = None
    best_D = 1
    
    # Try D=1 first with the known 5 points
    S1 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = S1
    best_D = 1
    
    # Try to find 6 points for small D
    for D in [2, 3, 5, 7, 11, 13]:
        # Simple greedy search
        # Start with (0,0)
        # Find points that are valid with (0,0)
        # Then find points valid with both, etc.
        
        # Limit search space
        limit = 50
        
        # Precompute valid differences
        # This is expensive, so we just check on the fly
        
        # Greedy construction
        S = [(0, 0)]
        for _ in range(5):
            found = False
            for y in range(-limit, limit + 1):
                for x in range(-limit, limit + 1):
                    P = (x, y)
                    if P in S:
                        continue
                    
                    valid = True
                    for Q in S:
                        d_sq = dist_sq(P, Q, D)
                        if not is_square(d_sq):
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Collinearity
                    for Q1, Q2 in combinations(S, 2):
                        if collinear(P, Q1, Q2):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Concyclicity
                    for Q1, Q2, Q3 in combinations(S, 3):
                        if concyclic(P, Q1, Q2, Q3, D):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    S.append(P)
                    found = True
                    break
                if found:
                    break
            
            if not found:
                break
        
        if len(S) > len(best_points):
            best_points = S
            best_D = D
            if len(S) >= 6:
                break # Found 6, good enough
        
    return {'D': best_D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
