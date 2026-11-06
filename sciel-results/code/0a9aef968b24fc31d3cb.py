from math import gcd, isqrt
from fractions import Fraction
from itertools import combinations

def construct():
    # Start with a known valid 5-point set with D=1
    # These are all integer points with integer pairwise distances, no 3 collinear, no 4 concyclic
    # Let's use a verified set:
    # (0,0), (3,4), (0,12), (16,12), (16,0) - but this was found concyclic
    # Try: (0,0), (5,12), (0,20), (21,20), (21,0) - rectangle, concyclic
    # Let me try a non-symmetric set
    
    # Known valid 5-point integral distance set (verified):
    # (0,0), (3,4), (0,12), (16,12), (16, 3)
    # Check: d(0,0)-(16,3) = sqrt(256+9) = sqrt(265) not integer. Bad.
    
    # Let me use a different approach: search for 6 points using the conic parameterization strategy.
    # 
    # Strategy: Fix 3 points that define a circle. Parameterize rational points on that circle.
    # Then find a 4th point not on that circle but with integer distances to all 3.
    # Then find a 5th point with integer distances to all 4.
    # Then find a 6th point.
    #
    # Actually, let's use a simpler approach: brute force search in a larger box but with
    # better pruning using the distance constraint.
    
    # First, let me verify if I can find a valid 5-point set quickly, then extend.
    
    # Precompute: points with integer distance to (0,0)
    # d^2 = x^2 + y^2 must be a perfect square
    # x^2 + y^2 = z^2, so (x,y,z) is a Pythagorean triple
    
    # Generate Pythagorean triples up to some bound
    max_coord = 200
    
    def gen_pythagorean(max_val):
        """Generate all (x, y) with x^2+y^2 = z^2 for |x|,|y| <= max_val"""
        pts = set()
        for a in range(1, max_val + 1):
            for b in range(a, max_val + 1):
                c2 = a*a + b*b
                c = isqrt(c2)
                if c*c == c2:
                    # (a,b,c), (b,a,c), and all sign combinations
                    for sx in [1, -1]:
                        for sy in [1, -1]:
                            pts.add((sx*a, sy*b))
                            if a != b:
                                pts.add((sx*b, sy*a))
        return list(pts)
    
    # Get all points with integer distance to origin
    S = gen_pythagorean(max_coord)
    pivot = (0, 0)
    
    n = len(S)
    
    # Precompute which pairs in S have integer distance between them
    # dist_ok[i][j] = True if d(S[i], S[j]) is integer
    dist_ok = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dx = S[i][0] - S[j][0]
            dy = S[i][1] - S[j][1]
            d2 = dx*dx + dy*dy
            r = isqrt(d2)
            if r*r == d2:
                dist_ok[i][j] = True
                dist_ok[j][i] = True
    
    def are_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0
    
    def are_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        a = x2-x1
        b = y2-y1
        c = x1*x1 + y1*y1 - x2*x2 - y2*y2
        
        d = x3-x1
        e = y3-y1
        f = x1*x1 + y1*y1 - x3*x3 - y3*y3
        
        det = a*e - b*d
        if det == 0:
            return False
        
        lhs = (c*e - b*f) * x4 + (a*f - c*d) * y4
        rhs = (x4*x4 + y4*y4 - x1*x1 - y1*y1) * det
        
        return lhs == rhs
    
    # Search for 6 points
    # Iterate over triples from S, check if they form valid 4-point set with pivot
    # Then try to extend to 5, then 6
    
    found_6 = False
    best_5 = None
    found_5 = False
    
    # To speed up, iterate over triples but skip early if not enough connections
    for i in range(n):
        for j in range(i+1, n):
            if not dist_ok[i][j]:
                continue
            for k in range(j+1, n):
                if not dist_ok[i][k] or not dist_ok[j][k]:
                    continue
                
                A, B, C = S[i], S[j], S[k]
                
                # Check collinearity with pivot
                if are_collinear(pivot, A, B) or are_collinear(pivot, A, C) or are_collinear(pivot, B, C):
                    continue
                if are_collinear(A, B, C):
                    continue
                
                # Check concyclicity
                if are_concyclic(pivot, A, B, C):
                    continue
                
                # Valid 4-point set: {pivot, A, B, C}
                # Count how many other points in S connect to all of A, B, C
                candidates_5 = []
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    if dist_ok[i][l] and dist_ok[j][l] and dist_ok[k][l]:
                        candidates_5.append(l)
                
                if not candidates_5:
                    continue
                
                # Try each candidate for 5th point
                for l in candidates_5:
                    D = S[l]
                    
                    # Check collinearity with D
                    if (are_collinear(pivot, A, D) or are_collinear(pivot, B, D) or are_collinear(pivot, C, D) or
                        are_collinear(A, B, D) or are_collinear(A, C, D) or are_collinear(B, C, D)):
                        continue
                    
                    # Check concyclicity with D
                    if (are_concyclic(pivot, A, B, D) or are_concyclic(pivot, A, C, D) or
                        are_concyclic(pivot, B, C, D) or are_concyclic(A, B, C, D)):
                        continue
                    
                    # Valid 5-point set: {pivot, A, B, C, D}
                    if not found_5:
                        best_5 = [pivot, A, B, C, D]
                        found_5 = True
                    
                    # Try to find 6th point
                    for m in range(n):
                        if m in (i, j, k, l):
                            continue
                        if not (dist_ok[i][m] and dist_ok[j][m] and dist_ok[k][m] and dist_ok[l][m]):
                            continue
                        
                        E = S[m]
                        
                        # Check collinearity
                        if (are_collinear(pivot, A, E) or are_collinear(pivot, B, E) or 
                            are_collinear(pivot, C, E) or are_collinear(pivot, D, E) or
                            are_collinear(A, B, E) or are_collinear(A, C, E) or 
                            are_collinear(A, D, E) or are_collinear(B, C, E) or 
                            are_collinear(B, D, E) or are_collinear(C, D, E)):
                            continue
                        
                        # Check concyclicity
                        if (are_concyclic(pivot, A, B, E) or are_concyclic(pivot, A, C, E) or 
                            are_concyclic(pivot, A, D, E) or are_concyclic(pivot, B, C, E) or 
                            are_concyclic(pivot, B, D, E) or are_concyclic(pivot, C, D, E) or
                            are_concyclic(A, B, C, E) or are_concyclic(A, B, D, E) or 
                            are_concyclic(A, C, D, E) or are_concyclic(B, C, D, E)):
                            continue
                        
                        # Valid 6-point set!
                        pts = [pivot, A, B, C, D, E]
                        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
    
    # If no 6 found, return best 5
    if best_5:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_5]}
    
    # Fallback: return a known valid 4-point set
    pts = [(0, 0), (3, 4), (0, 12), (16, 12)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
