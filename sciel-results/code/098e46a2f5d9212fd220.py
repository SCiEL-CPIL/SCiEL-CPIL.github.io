"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def _concyclic_det(p1, p2, p3, p4):
        """Exact 4x4 determinant for concyclic check using integer arithmetic."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Determinant of:
        # | x1 y1 x1^2+y1^2 1 |
        # | x2 y2 x2^2+y2^2 1 |
        # | x3 y3 x3^2+y3^2 1 |
        # | x4 y4 x4^2+y4^2 1 |
        
        # Subtract row 4 from rows 1, 2, 3 to simplify?
        # Or just compute directly.
        # Let's use the formula for determinant of 4x4.
        
        a1, b1, c1 = x1, y1, x1*x1 + y1*y1
        a2, b2, c2 = x2, y2, x2*x2 + y2*y2
        a3, b3, c3 = x3, y3, x3*x3 + y3*y3
        a4, b4, c4 = x4, y4, x4*x4 + y4*y4
        
        # Subtracting row 4 from others makes the last column [0,0,0,1] effectively if we expand?
        # Actually, let's just do the standard expansion or Bareiss.
        # Standard expansion is O(1) for fixed size.
        
        # det = a1(b2(c3*1 - c4*1) - b3(c2*1 - c4*1) + c3(b2*1 - b3*1)) ... wait, this is for 3x3 minor.
        
        # Let's use the property:
        # det = x1(y2(x3^2+y3^2 - x4^2-y4^2) - y3(x2^2+y2^2 - x4^2-y4^2) + (x3y2 - x2y3)(x4^2+y4^2)) ... this is getting messy.
        
        # Reliable 4x4 determinant:
        # | a b c d |
        # | e f g h |
        # | i j k l |
        # | m n o p |
        
        # We have columns: [x, y, x^2+y^2, 1]
        # Let's subtract row 4 from 1, 2, 3.
        # R1' = [x1-x4, y1-y4, x1^2+y1^2 - (x4^2+y4^2), 0]
        # R2' = [x2-x4, y2-y4, x2^2+y2^2 - (x4^2+y4^2), 0]
        # R3' = [x3-x4, y3-y4, x3^2+y3^2 - (x4^2+y4^2), 0]
        # R4 = [x4, y4, x4^2+y4^2, 1]
        
        # Det = 1 * Det of 3x3 matrix formed by R1', R2', R3' (first 3 cols)
        
        A = x1 - x4
        B = y1 - y4
        C = x1*x1 + y1*y1 - (x4*x4 + y4*y4)
        
        D = x2 - x4
        E = y2 - y4
        F = x2*x2 + y2*y2 - (x4*x4 + y4*y4)
        
        G = x3 - x4
        H = y3 - y4
        I = x3*x3 + y3*y3 - (x4*x4 + y4*y4)
        
        det3 = A*(E*I - F*H) - B*(D*I - F*G) + C*(D*H - E*G)
        return det3

    def all_constraints_satisfied(points):
        """Check all pairwise distances are integer, no 3 collinear, no 4 concyclic."""
        n = len(points)
        # Check all pairwise distances are integers
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        # Check no four concyclic
        for combo in combinations(range(n), 4):
            if _concyclic_det(points[combo[0]], points[combo[1]], 
                            points[combo[2]], points[combo[3]]) == 0:
                return False
        return True

    def generate_candidates(limit):
        """Generate a pool of candidate points."""
        cands = []
        for x in range(-limit, limit + 1):
            for y in range(-limit, limit + 1):
                cands.append((x, y))
        return cands

    def greedy_search(initial_points, candidates, max_points=12):
        """
        Greedily add points from candidates to initial_points.
        Returns the best set found.
        """
        current_set = list(initial_points)
        # Filter candidates to those not already in set
        current_set_tuples = set(current_set)
        avail_cands = [c for c in candidates if c not in current_set_tuples]
        
        # Shuffle candidates to avoid bias? Or sort by distance from centroid?
        # Let's shuffle for randomized restarts.
        random.shuffle(avail_cands)
        
        while len(current_set) < max_points:
            if elapsed() > DEADLINE:
                break
            added = False
            # We need to find a candidate that is valid with the current set.
            # To speed up, we can iterate through avail_cands.
            # If we find one, add it and restart the loop (remove from avail_cands).
            
            # Optimization: Check constraints incrementally.
            # For a candidate P, check:
            # 1. Integer distance to all Q in current_set.
            # 2. No 3 collinear in current_set U {P}.
            # 3. No 4 concyclic in current_set U {P}.
            
            n = len(current_set)
            # Pre-compute some things?
            
            for idx, cand in enumerate(avail_cands):
                # 1. Distance check
                valid_dist = True
                for q in current_set:
                    dx = cand[0] - q[0]
                    dy = cand[1] - q[1]
                    d2 = dx*dx + dy*dy
                    if not is_perfect_square(d2):
                        valid_dist = False
                        break
                if not valid_dist:
                    continue
                
                # 2. Collinearity check
                # Check if cand is collinear with any pair in current_set
                valid_coll = True
                for i in range(n):
                    for j in range(i+1, n):
                        if check_collinear(cand, current_set[i], current_set[j]):
                            valid_coll = False
                            break
                    if not valid_coll:
                        break
                if not valid_coll:
                    continue
                
                # 3. Concyclicity check
                # Check if cand is concyclic with any triple in current_set
                valid_conc = True
                for i in range(n):
                    for j in range(i+1, n):
                        for k in range(j+1, n):
                            if _concyclic_det(cand, current_set[i], current_set[j], current_set[k]) == 0:
                                valid_conc = False
                                break
                        if not valid_conc:
                            break
                    if not valid_conc:
                        break
                if not valid_conc:
                    continue
                
                # All checks passed
                current_set.append(cand)
                current_set_tuples.add(cand)
                avail_cands.pop(idx)
                added = True
                break
        
            if not added:
                break
        
        return current_set

    # Start with known good sets
    initial_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (4, 0), (0, 4)], # Check if valid
        [(0, 0), (5, 0), (0, 12), (5, 12)], # Check if valid
    ]
    
    # Verify initial sets
    valid_initials = []
    for s in initial_sets:
        if all_constraints_satisfied(s):
            valid_initials.append(s)
    
    if not valid_initials:
        # Fallback to a simple valid set
        valid_initials = [[(0,0), (3,4), (4,0)]]
        
    best_points = []
    
    # Generate a large candidate pool
    # Limit 100 might be too small for large sets, but 200 is 160k points.
    # Let's try 100 first, then 150 if time permits.
    limit = 100
    candidates = generate_candidates(limit)
    
    # Run multiple restarts
    num_restarts = 20
    
    for i in range(num_restarts):
        if elapsed() > DEADLINE - 5:
            break
            
        # Pick an initial set
        init_set = valid_initials[i % len(valid_initials)]
        
        # Run greedy search
        result = greedy_search(init_set, candidates, max_points=12)
        
        if len(result) > len(best_points):
            best_points = result
            
        # If we found a very large set, we can stop or continue to try for even larger
        if len(best_points) >= 12:
            break
            
    # If we didn't find anything better than the initial 5, try a larger limit or different strategy?
    # For now, return the best found.
    
    # Ensure best_points is not empty
    if not best_points:
        best_points = valid_initials[0]
        
    # Convert to required format with D=1
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best_points]

    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END
