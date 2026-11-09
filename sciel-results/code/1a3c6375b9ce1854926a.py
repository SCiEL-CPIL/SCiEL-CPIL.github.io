import itertools
import math
import random
from fractions import Fraction

def construct():
    # Approach: Use Pythagorean triples to guarantee integer distances.
    # Points are (x, y) with integer coordinates.
    # Distance between (x1,y1) and (x2,y2) is sqrt((dx)^2 + (dy)^2).
    # We need this to be integer, so (dx, dy, dist) must be a Pythagorean triple.
    
    # Strategy: Build points incrementally.
    # For each new point, check:
    # 1. All pairwise distances are integers (Pythagorean)
    # 2. No three collinear
    # 3. No four concyclic
    
    # Start with a known good seed and try to extend.
    
    # Known 5-point configuration (all integer distances, no 3 collinear, no 4 concyclic)
    seed = [(0, 0), (3, 4), (0, 5), (4, 3), (5, 0)]
    
    # Verify seed
    # (0,0)-(3,4): 5
    # (0,0)-(0,5): 5
    # (0,0)-(4,3): 5
    # (0,0)-(5,0): 5
    # (3,4)-(0,5): sqrt(9+1)=sqrt(10) - NOT integer!
    
    # Let me use a better seed.
    # Use points on a circle with rational distances? No, we need integer distances.
    
    # Better: use Pythagorean triple based construction.
    # Point A=(0,0). Points at distance d from A where d is integer.
    # (3,4): dist 5
    # (5,0): dist 5
    # (0,5): dist 5
    # (4,3): dist 5
    # But (3,4)-(5,0): sqrt(4+16)=sqrt(20) - not integer
    
    # Let me think differently. Use a grid and find maximal sets.
    
    # Precompute which pairs have integer distance
    N = 30
    points = [(x, y) for x in range(-N, N+1) for y in range(-N, N+1)]
    n = len(points)
    
    # Precompute integer distance pairs
    # dist_ok[i][j] = True if distance between points[i] and points[j] is integer
    dist_ok = [[False]*n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx*dx + dy*dy
            d = math.isqrt(d2)
            if d*d == d2:
                dist_ok[i][j] = True
                dist_ok[j][i] = True
    
    # Now do greedy/backtracking search for largest set
    # with no 3 collinear and no 4 concyclic
    
    best_set = []
    
    # Use a more structured approach: try to build from known small sets
    # and extend them.
    
    # Start with various 3-point seeds and try to extend
    random.seed(42)
    
    # Generate candidate 3-point sets with integer distances
    triplets = []
    for i in range(n):
        for j in range(i+1, n):
            if not dist_ok[i][j]:
                continue
            for k in range(j+1, n):
                if dist_ok[i][k] and dist_ok[j][k]:
                    p1, p2, p3 = points[i], points[j], points[k]
                    # Check not collinear
                    if not _collinear(p1, p2, p3):
                        triplets.append([i, j, k])
    
    # For each triplet, try to extend greedily
    # Sort triplets by some heuristic (e.g., spread)
    
    best_size = 0
    best_indices = []
    
    # Try a subset of triplets for speed
    if len(triplets) > 5000:
        random.shuffle(triplets)
        triplets = triplets[:5000]
    
    for triplet in triplets:
        current = list(triplet)
        
        # Try to extend
        for _ in range(100):  # max extensions to try per triplet
            # Find candidates that can be added
            candidates = []
            for idx in range(n):
                if idx in current:
                    continue
                # Check distance to all in current
                valid = True
                for ci in current:
                    if not dist_ok[idx][ci]:
                        valid = False
                        break
                if not valid:
                    continue
                # Check no 3 collinear
                p_new = points[idx]
                ok = True
                for a in range(len(current)):
                    for b in range(a+1, len(current)):
                        if _collinear(p_new, points[current[a]], points[current[b]]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # Check no 4 concyclic
                for a in range(len(current)):
                    for b in range(a+1, len(current)):
                        for c in range(b+1, len(current)):
                            if _concyclic(p_new, points[current[a]], points[current[b]], points[current[c]]):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append(idx)
            
            if not candidates:
                break
            
            # Pick a candidate (greedy: pick one that maximizes future options)
            # For simplicity, pick randomly or the first one
            random.shuffle(candidates)
            current.append(candidates[0])
        
        if len(current) > best_size:
            best_size = len(current)
            best_indices = current[:]
        
        # Early exit if we found 6+
        if best_size >= 6:
            break
    
    # If we didn't find a good set, use a known valid set
    if len(best_indices) < 3:
        # Fallback: simple valid 3-point set
        best_indices = [0, 1, 2]  # (0,0), (1,0), (0,1) - wait, need to check
        # Actually let me just hardcode a known valid set
        # (0,0), (3,4), (0,5): distances 5, 5, sqrt(9+1)=sqrt(10) - no
        # (0,0), (3,4), (5,0): distances 5, 5, sqrt(4+16)=sqrt(20) - no
        # (0,0), (4,3), (5,0): distances 5, 5, 1 - collinear? (0,0),(4,3),(5,0) - not collinear
        #   (0,0)-(4,3): 5, (0,0)-(5,0): 5, (4,3)-(5,0): sqrt(1+9)=sqrt(10) - no
        
        # Let me just use the original seed from the problem
        best_indices = []
        for i, p in enumerate([(0,0), (-56,33), (-16,63), (0,33), (16,63)]):
            best_indices.append(i)
    
    # Convert to output format
    result_points = []
    for idx in best_indices:
        x, y = points[idx]
        result_points.append([[x, 1], [y, 1]])
    
    return {'D': 1, 'points': result_points}

def _collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    d1 = x1*x1 + y1*y1
    d2 = x2*x2 + y2*y2
    d3 = x3*x3 + y3*y3
    d4 = x4*x4 + y4*y4
    
    # Determinant of [[x, y, x^2+y^2, 1] for each point]
    # If det == 0, they are concyclic (or collinear, but we check that separately)
    
    # Use exact integer arithmetic
    # det = |x1 y1 d1 1|
    #       |x2 y2 d2 1|
    #       |x3 y3 d3 1|
    #       |x4 y4 d4 1|
    
    # Subtract row 1 from rows 2,3,4 to simplify
    # |x1 y1 d1 1|
    # |x2-x1 y2-y1 d2-d1 0|
    # |x3-x1 y3-y1 d3-d1 0|
    # |x4-x1 y4-y1 d4-d1 0|
    
    # Now expand along last column:
    # det = 1 * det of 3x3:
    # |x2-x1 y2-y1 d2-d1|
    # |x3-x1 y3-y1 d3-d1|
    # |x4-x1 y4-y1 d4-d1|
    
    a = x2-x1; b = y2-y1; c = d2-d1
    d = x3-x1; e = y3-y1; f = d3-d1
    g = x4-x1; h = y4-y1; i = d4-d1
    
    det3 = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    
    return det3 == 0
