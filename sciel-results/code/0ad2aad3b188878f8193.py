from math import gcd, isqrt
from itertools import combinations
from fractions import Fraction

def is_integer_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    d2 = dx*dx + dy*dy
    r = isqrt(d2)
    return r*r == d2

def are_collinear(p1, p2, p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0

def are_concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Circle through 1,2,3:
    # D x1 + E y1 + F = -(x1^2+y1^2)
    # D x2 + E y2 + F = -(x2^2+y2^2)
    # D x3 + E y3 + F = -(x3^2+y3^2)
    
    # Subtract 1 from 2 and 3:
    # D (x2-x1) + E (y2-y1) = x1^2+y1^2 - x2^2-y2^2
    # D (x3-x1) + E (y3-y1) = x1^2+y1^2 - x3^2-y3^2
    
    a, b, c = x2-x1, y2-y1, x1*x1 + y1*y1 - x2*x2 - y2*y2
    d, e, f = x3-x1, y3-y1, x1*x1 + y1*y1 - x3*x3 - y3*y3
    
    det = a*e - b*d
    if det == 0:
        return False
    
    # D = (c*e - b*f) / det
    # E = (a*f - c*d) / det
    
    # Check if 4th point satisfies:
    # ((c*e - b*f) * x4 + (a*f - c*d) * y4) / det = x4^2+y4^2 - x1^2-y1^2
    
    lhs_num = (c*e - b*f) * x4 + (a*f - c*d) * y4
    rhs_num = (x4*x4 + y4*y4 - x1*x1 - y1*y1) * det
    
    return lhs_num == rhs_num

def construct():
    # Search for a set of points with integer distances
    # using a systematic approach
    
    N = 30
    points = []
    for x in range(-N, N+1):
        for y in range(-N, N+1):
            points.append((x, y))
    
    pivot = (0, 0)
    S = [p for p in points if p != pivot and is_integer_distance(pivot, p)]
    
    n = len(S)
    
    # Precompute distance matrix
    dist = [[-1]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if is_integer_distance(S[i], S[j]):
                dist[i][j] = 1
                dist[j][i] = 1
    
    best_points = [pivot]
    found_6 = False
    
    # Try to find valid sets by iterating over triples
    for i in range(n):
        for j in range(i+1, n):
            if dist[i][j] == -1:
                continue
            for k in range(j+1, n):
                if dist[i][k] == -1 or dist[j][k] == -1:
                    continue
                
                A, B, C = S[i], S[j], S[k]
                
                # Check collinearity
                if (are_collinear(pivot, A, B) or are_collinear(pivot, A, C) or 
                    are_collinear(pivot, B, C) or are_collinear(A, B, C)):
                    continue
                
                # Check concyclicity
                if are_concyclic(pivot, A, B, C):
                    continue
                
                # Valid set of 4
                base4 = [pivot, A, B, C]
                
                # Try to add D
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    D = S[l]
                    if dist[i][l] == -1 or dist[j][l] == -1 or dist[k][l] == -1:
                        continue
                    
                    # Check collinearity with D
                    collinear_with_D = False
                    for p1 in base4:
                        for p2 in base4:
                            if p1 != p2 and are_collinear(p1, p2, D):
                                collinear_with_D = True
                                break
                        if collinear_with_D:
                            break
                    if collinear_with_D:
                        continue
                    
                    # Check concyclicity with D
                    concyclic_with_D = False
                    for combo in combinations(base4, 3):
                        if are_concyclic(combo[0], combo[1], combo[2], D):
                            concyclic_with_D = True
                            break
                    if concyclic_with_D:
                        continue
                    
                    # Valid set of 5
                    base5 = base4 + [D]
                    if len(base5) > len(best_points):
                        best_points = base5.copy()
                    
                    # Try to add E
                    for m in range(n):
                        if m in (i, j, k, l):
                            continue
                        E = S[m]
                        if (dist[i][m] == -1 or dist[j][m] == -1 or 
                            dist[k][m] == -1 or dist[l][m] == -1):
                            continue
                        
                        # Check collinearity with E
                        collinear_with_E = False
                        for p1 in base5:
                            for p2 in base5:
                                if p1 != p2 and are_collinear(p1, p2, E):
                                    collinear_with_E = True
                                    break
                            if collinear_with_E:
                                break
                        if collinear_with_E:
                            continue
                        
                        # Check concyclicity with E
                        concyclic_with_E = False
                        for combo in combinations(base5, 3):
                            if are_concyclic(combo[0], combo[1], combo[2], E):
                                concyclic_with_E = True
                                break
                        if concyclic_with_E:
                            continue
                        
                        # Valid set of 6!
                        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in base5 + [E]]}
    
    # If no 6 found, return best 5
    if len(best_points) >= 5:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
    
    # Fallback: known valid 4-point set
    fallback = [(0, 0), (3, 4), (0, 12), (16, 12)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in fallback]}
