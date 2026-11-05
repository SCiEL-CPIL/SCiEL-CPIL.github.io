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
        current_set_tuples = set(current_set)
        avail_cands = [c for c in candidates if c not in current_set_tuples]
        
        # Shuffle candidates to avoid bias
        random.shuffle(avail_cands)
        
        while len(current_set) < max_points:
            if elapsed() > DEADLINE:
                break
            added = False
            n = len(current_set)
            
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

    def local_search(current_set, candidates, max_points=12, max_iterations=100):
        """
        Local search: try to swap a point in current_set with a candidate to improve.
        This helps escape local optima.
        """
        current_set = list(current_set)
        current_set_tuples = set(current_set)
        
        for _ in range(max_iterations):
            if elapsed() > DEADLINE:
                break
            if len(current_set) >= max_points:
                break
            
            # Try to add a new point
            added = False
            for cand in candidates:
                if cand in current_set_tuples:
                    continue
                
                # Check if cand can be added
                valid = True
                for q in current_set:
                    dx = cand[0] - q[0]
                    dy = cand[1] - q[1]
                    d2 = dx*dx + dy*dy
                    if not is_perfect_square(d2):
                        valid = False
                        break
                if not valid:
                    continue
                
                valid_coll = True
                n = len(current_set)
                for i in range(n):
                    for j in range(i+1, n):
                        if check_collinear(cand, current_set[i], current_set[j]):
                            valid_coll = False
                            break
                    if not valid_coll:
                        break
                if not valid_coll:
                    continue
                
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
                
                # Can add
                current_set.append(cand)
                current_set_tuples.add(cand)
                added = True
                break
            
            if added:
                continue
            
            # Try swap: remove one point, add another
            swapped = False
            for i in range(len(current_set)):
                if elapsed() > DEADLINE:
                    break
                removed = current_set[i]
                temp_set = current_set[:i] + current_set[i+1:]
                temp_tuples = set(temp_set)
                
                # Try to add a candidate
                for cand in candidates:
                    if cand in temp_tuples:
                        continue
                    
                    valid = True
                    for q in temp_set:
                        dx = cand[0] - q[0]
                        dy = cand[1] - q[1]
                        d2 = dx*dx + dy*dy
                        if not is_perfect_square(d2):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    valid_coll = True
                    n = len(temp_set)
                    for a in range(n):
                        for b in range(a+1, n):
                            if check_collinear(cand, temp_set[a], temp_set[b]):
                                valid_coll = False
                                break
                        if not valid_coll:
                            break
                    if not valid_coll:
                        continue
                    
                    valid_conc = True
                    for a in range(n):
                        for b in range(a+1, n):
                            for c in range(b+1, n):
                                if _concyclic_det(cand, temp_set[a], temp_set[b], temp_set[c]) == 0:
                                    valid_conc = False
                                    break
                            if not valid_conc:
                                break
                        if not valid_conc:
                            break
                    if not valid_conc:
                        continue
                    
                    # Swap succeeded
                    current_set = temp_set + [cand]
                    current_set_tuples = set(current_set)
                    swapped = True
                    break
                
                if swapped:
                    break
            
            if not swapped:
                break
        
        return current_set

    # Start with known good sets
    initial_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (4, 0), (0, 4)],
        [(0, 0), (5, 0), (0, 12), (5, 12)],
    ]
    
    # Verify initial sets
    valid_initials = []
    for s in initial_sets:
        if all_constraints_satisfied(s):
            valid_initials.append(s)
    
    if not valid_initials:
        valid_initials = [[(0,0), (3,4), (4,0)]]
        
    best_points = []
    
    # Generate candidate pools with different limits
    limits = [80, 120, 160]
    
    for limit in limits:
        if elapsed() > DEADLINE - 10:
            break
        
        candidates = generate_candidates(limit)
        
        # Run multiple restarts with different initial sets
        num_restarts = 15
        for i in range(num_restarts):
            if elapsed() > DEADLINE - 5:
                break
            
            init_set = valid_initials[i % len(valid_initials)]
            
            # Run greedy search
            result = greedy_search(init_set, candidates, max_points=12)
            
            # Try local search to improve
            if len(result) < 12:
                result = local_search(result, candidates, max_points=12, max_iterations=50)
            
            if len(result) > len(best_points):
                best_points = result
            
            if len(best_points) >= 12:
                break
        
        if len(best_points) >= 12:
            break
    
    # Ensure best_points is not empty
    if not best_points:
        best_points = valid_initials[0]
        
    # Convert to required format with D=1
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best_points]

    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END
