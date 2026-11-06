import math
import time
import random
from itertools import combinations

def is_square(n):
    if n < 0: return False
    r = math.isqrt(n)
    return r*r == n

def dist_sq(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def is_integer_dist(p1, p2):
    return is_square(dist_sq(p1, p2))

def collinear(p1, p2, p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0

def concyclic(p1, p2, p3, p4):
    # Determinant of circle equation
    # x^2 + y^2 + Dx + Ey + F = 0
    # | x1^2+y1^2  x1  y1  1 |
    # | x2^2+y2^2  x2  y2  1 |
    # | x3^2+y3^2  x3  y3  1 | = 0
    # | x4^2+y4^2  x4  y4  1 |
    def det4(m):
        # simple 4x4 determinant
        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
             - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
             + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
             - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
    pts = [p1, p2, p3, p4]
    m = [[p[0]**2 + p[1]**2, p[0], p[1], 1] for p in pts]
    return det4(m) == 0

def construct():
    start_time = time.time()
    deadline = 130.0 # 20s margin
    
    # Try to find a set of size 6 or 7
    best_set = [(0,0), (-56, 33), (-16, 63), (0, 33), (16, 63)] # Fallback from seed
    
    # Search for larger sets
    # We'll use a randomized local search / backtracking
    # For efficiency, we restrict coordinates to a reasonable box
    # and use symmetry breaking.
    
    # Strategy: 
    # 1. Fix P0=(0,0), P1=(1,0)
    # 2. Search for P2 with y>0, integer dist to P0, P1
    # 3. Search for P3, P4, P5...
    
    # To make it faster, we can pre-compute a list of candidate points
    # that have integer distances to (0,0) and (1,0).
    # dist to (0,0): x^2+y^2 = a^2
    # dist to (1,0): (x-1)^2+y^2 = b^2
    # Subtracting: (x-1)^2 - x^2 = b^2 - a^2 => -2x + 1 = b^2 - a^2 => 2x = 1 - b^2 + a^2
    # So x must be integer.
    
    max_coord = 200
    candidates = []
    for x in range(-max_coord, max_coord+1):
        for y in range(1, max_coord+1): # y>0 for P2
            d0_sq = x*x + y*y
            d1_sq = (x-1)*(x-1) + y*y
            if is_square(d0_sq) and is_square(d1_sq):
                candidates.append((x, y))
                
    # Shuffle candidates to avoid bias
    random.shuffle(candidates)
    
    # Backtracking search
    def search(current_set, target_size):
        if len(current_set) == target_size:
            return current_set[:]
        if time.time() > deadline:
            return None
            
        # Generate next candidates
        # We can filter candidates based on distance to all points in current_set
        # To speed up, we can just iterate through a subset of candidates
        # and check constraints.
        
        # For efficiency, we pick a random candidate from the pool
        # and check if it fits. If not, try another.
        # This is a randomized local search.
        
        # A better approach: Iterate through candidates, check if it can be added.
        # If yes, recurse.
        
        # To prevent explosion, we limit the number of candidates we try at each step.
        # Since we have a deadline, we can just loop.
        
        # Let's try a simpler greedy approach with random restarts for the whole set.
        # Actually, backtracking is better for finding larger sets.
        
        # Optimization: Pre-check collinearity with pairs in current_set
        # and concyclicity with triples in current_set.
        
        for p in candidates:
            # Check distance to all in current_set
            valid = True
            for q in current_set:
                if not is_integer_dist(p, q):
                    valid = False
                    break
            if not valid:
                continue
                
            # Check collinearity
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    if collinear(p, current_set[i], current_set[j]):
                        valid = False
                        break
                if not valid: break
            if not valid:
                continue
                
            # Check concyclicity
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    for k in range(j+1, len(current_set)):
                        if concyclic(p, current_set[i], current_set[j], current_set[k]):
                            valid = False
                            break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid:
                continue
                
            # Add p and recurse
            res = search(current_set + [p], target_size)
            if res:
                return res
            if time.time() > deadline:
                return None
                
        return None

    # Try to find size 6
    # Fix P0, P1
    base = [(0,0), (1,0)]
    # We need P2 from candidates
    # But search() above assumes current_set has 2 points.
    # Let's modify search to handle the base case.
    
    # Re-implement search to be more robust
    def search_from(base, target_size):
        if len(base) >= target_size:
            return base[:]
        if time.time() > deadline:
            return None
            
        # Try to add points from candidates
        # To improve success rate, we can randomize the candidate order
        # or use a different strategy.
        
        # For now, iterate through candidates
        # To avoid getting stuck in a local minimum, we can shuffle candidates
        # before each call, but that's expensive.
        
        # Instead, we'll just iterate.
        for p in candidates:
            # Skip if p is already in base (shouldn't happen with unique candidates)
            if p in base:
                continue
                
            # Check constraints
            valid = True
            for q in base:
                if not is_integer_dist(p, q):
                    valid = False
                    break
            if not valid: continue
            
            for i in range(len(base)):
                for j in range(i+1, len(base)):
                    if collinear(p, base[i], base[j]):
                        valid = False
                        break
                if not valid: break
            if not valid: continue
            
            for i in range(len(base)):
                for j in range(i+1, len(base)):
                    for k in range(j+1, len(base)):
                        if concyclic(p, base[i], base[j], base[k]):
                            valid = False
                            break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: continue
            
            # Recurse
            res = search_from(base + [p], target_size)
            if res:
                return res
            if time.time() > deadline:
                return None
                
        return None

    # Try sizes 6, 7
    for target in [6, 7]:
        # We can try different base points or just use the fixed base
        # Fixed base: (0,0), (1,0)
        res = search_from([(0,0), (1,0)], target)
        if res:
            best_set = res
            # If we found 7, we can break. If we found 6, we might try to find 7 later
            # but for now, 6 is better than 5.
            if target == 7:
                break
        if time.time() > deadline:
            break
            
    # Convert to output format
    # D=1
    # points: [[x_num, x_den], [y_num, y_den]]
    out_points = []
    for p in best_set:
        out_points.append([[p[0], 1], [p[1], 1]])
        
    return {'D': 1, 'points': out_points}
