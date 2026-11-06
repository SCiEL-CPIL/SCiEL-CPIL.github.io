from math import gcd, isqrt
from itertools import combinations

def is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def dist2_int(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    d2 = dx * dx + dy * dy
    return d2

def is_integer_distance(p1, p2):
    d2 = dist2_int(p1, p2)
    return is_square(d2)

def are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def are_concyclic(p1, p2, p3, p4):
    # Check if 4 points are concyclic using determinant
    # | x   y   x^2+y^2   1 |
    # | x   y   x^2+y^2   1 | = 0
    # | x   y   x^2+y^2   1 |
    # | x   y   x^2+y^2   1 |
    
    # Subtract row 1 from rows 2, 3, 4 to simplify
    # Let r_i = (x_i, y_i, x_i^2+y_i^2, 1)
    # We can compute the determinant of the 3x3 matrix formed by differences.
    # Or just use the formula:
    # (x2-x1)^2 + (y2-y1)^2 = ...
    
    # A more stable way:
    # Circle eq: x^2 + y^2 + Ax + By + C = 0
    # x1^2 + y1^2 + A x1 + B y1 + C = 0
    # x2^2 + y2^2 + A x2 + B y2 + C = 0
    # x3^2 + y3^2 + A x3 + B y3 + C = 0
    # x4^2 + y4^2 + A x4 + B y4 + C = 0
    
    # Subtract 1 from 2,3,4:
    # A(x2-x1) + B(y2-y1) = (x1^2+y1^2) - (x2^2+y2^2)
    # A(x3-x1) + B(y3-y1) = (x1^2+y1^2) - (x3^2+y3^2)
    # A(x4-x1) + B(y4-y1) = (x1^2+y1^2) - (x4^2+y4^2)
    
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = x1*x1 + y1*y1 - x2*x2 - y2*y2
    
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = x1*x1 + y1*y1 - x3*x3 - y3*y3
    
    a3 = x4 - x1
    b3 = y4 - y1
    c3 = x1*x1 + y1*y1 - x4*x4 - y4*y4
    
    # Determinant of [[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]] must be 0
    det = a1 * (b2 * c3 - c2 * b3) - b1 * (a2 * c3 - c2 * a3) + c1 * (a2 * b3 - b2 * a3)
    return det == 0

def construct():
    # Generate candidates
    # We want points (x,y) such that dist to (0,0) is integer.
    # And we want a good mix.
    M = 30
    candidates = [(0,0)]
    for x in range(-M, M+1):
        for y in range(-M, M+1):
            if x == 0 and y == 0: continue
            if is_square(x*x + y*y):
                candidates.append((x,y))
    
    # Remove duplicates (shouldn't be any, but just in case)
    candidates = list(set(candidates))
    
    # Sort candidates to prioritize those with many connections?
    # Or just randomize/shuffle to avoid local minima?
    # Let's stick to a deterministic order for reproducibility, but maybe
    # prioritize points that are "far" from each other to avoid concyclicity?
    
    # Backtracking search
    best_set = []
    
    def backtrack(current_set, remaining):
        nonlocal best_set
        if len(current_set) > len(best_set):
            best_set = current_set[:]
        if not remaining:
            return
            
        # Try to add the first point from remaining
        # This is inefficient. We should try all points in remaining.
        
        for i in range(len(remaining)):
            p = remaining[i]
            # Check if p can be added to current_set
            valid = True
            for q in current_set:
                if not is_integer_distance(p, q):
                    valid = False
                    break
                for r in current_set:
                    if r == q: continue
                    if are_collinear(p, q, r):
                        valid = False
                        break
                if not valid: break
            if not valid: continue
            
            # Check concyclicity with any 3 in current_set
            if len(current_set) >= 3:
                for comb in combinations(current_set, 3):
                    if are_concyclic(p, comb[0], comb[1], comb[2]):
                        valid = False
                        break
            if not valid: continue
            
            # Add p
            new_set = current_set + [p]
            # Recurse
            # To speed up, we can prune.
            backtrack(new_set, remaining[i+1:])
            
            # Early stop if we found a very large set?
            if len(best_set) >= 7:
                return

    # To make the search faster, we can limit the depth or use heuristics.
    # The number of candidates is ~ 100-200.
    # Backtracking on 100 items is too slow if we don't prune well.
    
    # Let's try a different approach:
    # 1. Find a valid set of 4.
    # 2. Extend to 5, 6.
    
    # Actually, the previous grid search was O(N^4).
    # Let's stick to a modified grid search but with better pruning and bug fixes.
    # The "new design" requirement is tricky if I just fix bugs.
    # I will implement the "Candidate Generation" approach but limit the candidate set size
    # and use a greedy + local search hybrid.
    
    # Greedy:
    # 1. Start with (0,0).
    # 2. Pick next point that maximizes some metric?
    
    # Let's go with a randomized restart approach.
    import random
    random.seed(42)
    
    best_global = []
    
    for restart in range(20):
        # Shuffle candidates
        cands = candidates[:]
        random.shuffle(cands)
        
        current = [(0,0)]
        
        for p in cands:
            if p == (0,0): continue
            # Check if p can be added
            valid = True
            for q in current:
                if not is_integer_distance(p, q):
                    valid = False
                    break
            if not valid: continue
            
            # Check collinearity
            if len(current) >= 2:
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        if are_collinear(p, current[i], current[j]):
                            valid = False
                            break
                    if not valid: break
            if not valid: continue
            
            # Check concyclicity
            if len(current) >= 3:
                for comb in combinations(current, 3):
                    if are_concyclic(p, comb[0], comb[1], comb[2]):
                        valid = False
                        break
            if not valid: continue
            
            current.append(p)
            if len(current) > len(best_global):
                best_global = current[:]
                
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_global]}
