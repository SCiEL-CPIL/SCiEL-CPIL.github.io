import time
import random
from math import isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Helper to check if a number is a perfect square
    def is_square(n):
        if n < 0:
            return False
        s = isqrt(n)
        return s * s == n

    # Helper to check collinearity of three points (integers)
    def not_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]) != 0

    # Helper to check if four points are concyclic
    def not_concyclic(p1, p2, p3, p4):
        # Use determinant of 4x4 matrix
        # | x  y  x^2+y^2  1 |
        def det4(m):
            a, b, c, d = m[0]
            e, f, g, h = m[1]
            i, j, k, l = m[2]
            m4, n, o, p = m[3]
            def det3(sub):
                s1, s2, s3 = sub[0]
                t1, t2, t3 = sub[1]
                u1, u2, u3 = sub[2]
                return s1*(t2*u3 - t3*u2) - s2*(t1*u3 - t3*u1) + s3*(t1*u2 - t2*u1)
            return (a * det3([[f,g,h],[j,k,l],[n,o,p]]) - 
                    b * det3([[e,g,h],[i,k,l],[m4,o,p]]) + 
                    c * det3([[e,f,h],[i,j,l],[m4,n,p]]) - 
                    d * det3([[e,f,g],[i,j,k],[m4,n,o]]))
        
        matrix = [[x, y, x*x + y*y, 1] for x, y in [p1, p2, p3, p4]]
        return det4(matrix) != 0

    # Check if a point p can be added to set S
    def can_add(p, S):
        if p in S:
            return False
        for q in S:
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            dist_sq = dx*dx + dy*dy
            if not is_square(dist_sq):
                return False
        for q, r in combinations(S, 2):
            if not not_collinear(p, q, r):
                return False
        for q, r, s in combinations(S, 3):
            if not not_concyclic(p, q, r, s):
                return False
        return True

    # Generate candidate points with rational coordinates
    # For D=1, we need integer points with integer distances
    # Use Pythagorean triples to generate candidates
    max_coord = 150
    candidates = []
    
    # Add a base set of known rational distance points
    # Known 5-point rational distance set (scaled)
    base_points = [
        (0, 0),
        (3, 4),
        (0, 5),
        (4, 3),
        (5, 0),
        (-3, 4),
        (3, -4),
        (-4, 3),
        (4, -3),
        (-5, 0),
        (0, -5),
        (5, 5),
        (-5, 5),
        (5, -5),
        (-5, -5),
        (7, 24),
        (24, 7),
        (15, 8),
        (8, 15),
        (20, 21),
        (21, 20),
        (9, 12),
        (12, 9),
        (12, 16),
        (16, 12),
        (5, 12),
        (12, 5),
        (9, 40),
        (40, 9),
        (11, 60),
        (60, 11),
        (28, 45),
        (45, 28),
        (33, 56),
        (56, 33),
        (13, 84),
        (84, 13),
        (36, 77),
        (77, 36),
        (19, 180),
        (180, 19),
        (15, 112),
        (112, 15),
        (48, 55),
        (55, 48),
        (65, 72),
        (72, 65),
        (33, 112),
        (112, 33),
        (11, 120),
        (120, 11),
        (60, 91),
        (91, 60),
        (17, 144),
        (144, 17),
        (48, 105),
        (105, 48),
        (64, 93),
        (93, 64),
        (35, 120),
        (120, 35),
        (16, 105),
        (105, 16),
        (33, 156),
        (156, 33),
        (11, 168),
        (168, 11),
        (45, 168),
        (168, 45),
        (24, 187),
        (187, 24),
        (77, 156),
        (156, 77),
        (88, 105),
        (105, 88),
        (17, 168),
        (168, 17),
        (19, 180),
        (180, 19),
        (57, 176),
        (176, 57),
        (123, 160),
        (160, 123),
    ]
    
    for x, y in base_points:
        if abs(x) <= max_coord and abs(y) <= max_coord:
            candidates.append((x, y))
            candidates.append((-x, y))
            candidates.append((x, -y))
            candidates.append((-x, -y))
    
    # Add more candidates from Pythagorean triples
    for a in range(1, max_coord + 1):
        for b in range(a + 1, max_coord + 1):
            c2 = a*a + b*b
            c = isqrt(c2)
            if c * c == c2 and c <= max_coord:
                candidates.append((a, b))
                candidates.append((-a, b))
                candidates.append((a, -b))
                candidates.append((-a, -b))
                candidates.append((b, a))
                candidates.append((-b, a))
                candidates.append((b, -a))
                candidates.append((-b, -a))
    
    # Remove duplicates
    candidates = list(set(candidates))
    
    # Sort by distance from origin for more systematic search
    candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Try to build from known good seeds
    seed_sets = [
        [(0, 0), (3, 4), (0, 5), (4, 3), (5, 0)],
        [(0, 0), (3, 4), (4, 3), (5, 0), (3, 0)],
        [(0, 0), (5, 0), (0, 5), (3, 4), (4, 3)],
        [(0, 0), (3, 4), (0, 5), (5, 0), (-3, 4)],
    ]
    
    for seed in seed_sets:
        if time.time() > deadline:
            break
        
        current_set = list(seed)
        
        # Greedy expansion - try all candidates
        for p in candidates:
            if time.time() > deadline:
                break
            if p not in current_set and can_add(p, current_set):
                current_set.append(p)
                if len(current_set) > best_size:
                    best_set = list(current_set)
                    best_size = len(current_set)
                if best_size >= 8:
                    break
        
        if best_size >= 8:
            break
    
    # Strategy 2: Randomized restarts with different seeds
    if best_size < 7:
        random.shuffle(candidates)
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Pick a random starting configuration
            if restart < 10:
                # Start with a small random set
                start_idx = random.randint(0, len(candidates) - 1)
                current_set = [candidates[start_idx]]
                # Add a few more random points
                for _ in range(2):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, len(candidates) - 1)
                    if candidates[idx] not in current_set and can_add(candidates[idx], current_set):
                        current_set.append(candidates[idx])
            else:
                # Start with a random 3-point set
                current_set = random.sample(candidates, 3)
                # Verify it's valid
                valid = True
                for p, q, r in combinations(current_set, 3):
                    if not not_collinear(p, q, r):
                        valid = False
                        break
                if not valid:
                    continue
                # Check distances
                for p, q in combinations(current_set, 2):
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    if not is_square(dx*dx + dy*dy):
                        valid = False
                        break
                if not valid:
                    continue
            
            # Greedy expansion
            for p in candidates:
                if time.time() > deadline:
                    break
                if p not in current_set and can_add(p, current_set):
                    current_set.append(p)
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_size = len(current_set)
                    if best_size >= 8:
                        break
            
            if best_size >= 8:
                break
    
    # Strategy 3: If we have a good set, try local search to improve
    if best_size >= 4 and best_size < 8:
        # Try replacing one point with another to enable more additions
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            original = best_set[i]
            for p in candidates:
                if time.time() > deadline:
                    break
                if p == original:
                    continue
                new_set = best_set[:i] + [p] + best_set[i+1:]
                # Check if new_set is valid
                valid = True
                for p2, q2, r2 in combinations(new_set, 3):
                    if not not_collinear(p2, q2, r2):
                        valid = False
                        break
                if valid:
                    for p2, q2, r2, s2 in combinations(new_set, 4):
                        if not not_concyclic(p2, q2, r2, s2):
                            valid = False
                            break
                if valid:
                    for p2, q2 in combinations(new_set, 2):
                        dx = p2[0] - q2[0]
                        dy = p2[1] - q2[1]
                        if not is_square(dx*dx + dy*dy):
                            valid = False
                            break
                if valid:
                    # Try to expand new_set
                    current_set = list(new_set)
                    for p2 in candidates:
                        if time.time() > deadline:
                            break
                        if p2 not in current_set and can_add(p2, current_set):
                            current_set.append(p2)
                            if len(current_set) > best_size:
                                best_set = list(current_set)
                                best_size = len(current_set)
                            if best_size >= 8:
                                break
                    if best_size >= 8:
                        break
    
    # Format the result
    result_points = []
    for x, y in best_set:
        result_points.append([[x, 1], [y, 1]])
        
    return {'D': 1, 'points': result_points}

# EVOLVE-BLOCK-END
