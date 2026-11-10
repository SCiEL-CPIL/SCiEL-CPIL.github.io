import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    def evaluate(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def validate(coeffs):
        if len(coeffs) != 12:
            return False
        if coeffs[-1] == 0:
            return False
        if coeffs[0] == 0:
            return False
        if not all(isinstance(c, int) for c in coeffs):
            return False
        if not all(-64 <= c <= 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    random.seed(42)

    # Constraint satisfaction approach:
    # For a 12-term polynomial with consecutive exponents 0..11, the square has 23 terms.
    # We want to maximize cancellations (zero coefficients in the square).
    # The square coefficient at degree k is sum_{i+j=k} c[i]*c[j] for i,j in 0..11.
    # We can formulate this as: for each k where we want sq[k]=0, we have a linear constraint.
    
    # Precompute the convolution structure
    # sq[k] = sum over valid (i,j) with i+j=k of c[i]*c[j]
    # For k=0: c[0]^2
    # For k=1: 2*c[0]*c[1]
    # For k=2: 2*c[0]*c[2] + c[1]^2
    # ...
    # For k=11: 2*sum_{i=0..5} c[i]*c[11-i] + c[5.5]... wait, let me be more careful
    
    # Let's build the constraint matrix
    # c is a vector of 12 elements
    # sq[k] for k=0..22
    # We want sq[k] = 0 for as many k as possible
    
    # For a given target pattern of which sq[k] should be zero,
    # we can try to solve the system
    
    # Key insight: if we fix the support (which c[i] are nonzero), 
    # and try to find values that make certain sq[k] zero,
    # we get a system of quadratic equations
    
    # Alternative: use the fact that for small coefficient ranges,
    # we can do a structured search
    
    # Let's try a different approach: 
    # 1. Try small coefficient values first
    # 2. Use the structure of the convolution to guide the search
    
    # Precompute which pairs contribute to each degree
    pair_to_degree = {}
    for i in range(12):
        for j in range(12):
            k = i + j
            if k not in pair_to_degree:
                pair_to_degree[k] = []
            pair_to_degree[k].append((i, j))
    
    # For each degree k, sq[k] = sum of c[i]*c[j] for (i,j) in pair_to_degree[k]
    # Note: if i != j, both (i,j) and (j,i) contribute, so it's 2*c[i]*c[j]
    
    # Let's try a systematic search with pruning
    # We'll assign coefficients one by one, and at each step check
    # if any fully-determined sq[k] is already nonzero (prune)
    
    # Order of assignment: try to assign coefficients that affect the most
    # low-degree terms first, so we can prune early
    
    # Actually, let's try a smarter approach:
    # Start with a known good pattern and refine
    
    # Known: [1,1,1,1,1,1,1,1,1,1,1,1] gives square with 23 nonzero terms
    # We want to find cancellations
    
    # Let's try coefficients from a small set: {-2,-1,1,2}
    # and use backtracking with constraint propagation
    
    def backtrack_search(candidates, assigned, depth, current_sq):
        """
        candidates: list of possible values for each position
        assigned: dict mapping position -> value
        depth: current depth
        current_sq: list of 23 values, some may be partial sums
        """
        nonlocal best_score, best_poly
        
        if time.time() > deadline - 3:
            return
        
        if depth == 12:
            # Check all constraints
            sq = [0] * 23
            for i in range(12):
                for j in range(12):
                    sq[i+j] += assigned[i] * assigned[j]
            score = sum(1 for x in sq if x != 0)
            if score < best_score:
                best_score = score
                best_poly = [assigned[i] for i in range(12)]
            return
        
        # Try each candidate value for position depth
        for val in candidates[depth]:
            if val == 0:
                continue
            
            # Check if this assignment violates any fully-determined constraint
            # A constraint sq[k] is fully determined if all pairs (i,j) with i+j=k
            # have both i and j already assigned
            valid = True
            for k in range(23):
                pairs = pair_to_degree[k]
                # Check if all pairs are determined
                all_determined = True
                for (i, j) in pairs:
                    if i not in assigned and i != depth:
                        all_determined = False
                        break
                    if j not in assigned and j != depth:
                        all_determined = False
                        break
                if all_determined:
                    # Compute sq[k] with current assignment
                    sq_k = 0
                    for (i, j) in pairs:
                        ci = assigned.get(i, val if i == depth else None)
                        cj = assigned.get(j, val if j == depth else None)
                        if ci is None or cj is None:
                            valid = False
                            break
                        sq_k += ci * cj
                    if sq_k != 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            assigned[depth] = val
            backtrack_search(candidates, assigned, depth + 1, current_sq)
            del assigned[depth]
    
    # Define candidate sets for each position
    # Use small values for better chance of cancellation
    small_values = [-2, -1, 1, 2]
    medium_values = [-3, -2, -1, 1, 2, 3]
    
    # Try different candidate set sizes
    for val_range in [small_values, medium_values]:
        if time.time() > deadline - 10:
            break
        
        candidates = [val_range for _ in range(12)]
        backtrack_search(candidates, {}, 0, [0]*23)
    
    # Also try a randomized local search as fallback
    if best_poly is None or best_score > 15:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            curr = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            curr_score = evaluate(curr)
            if curr_score < best_score:
                best_score = curr_score
                best_poly = curr.copy()
    
    # Final polish: local search around best found
    if best_poly is not None and time.time() < deadline - 5:
        curr = best_poly.copy()
        curr_score = best_score
        
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            idx = random.randint(0, 11)
            old_val = curr[idx]
            
            candidates = set()
            for dv in [-3, -2, -1, 1, 2, 3]:
                nv = old_val + dv
                if -64 <= nv <= 64 and nv != 0:
                    candidates.add(nv)
            candidates.add(-old_val)
            candidates.add(random.choice([-3, -2, -1, 1, 2, 3]))
            
            improved = False
            for new_val in candidates:
                if new_val == 0 or new_val == old_val:
                    continue
                if not (-64 <= new_val <= 64):
                    continue
                
                curr[idx] = new_val
                new_score = evaluate(curr)
                
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr.copy()
                        no_improve = 0
                    break
            if not improved:
                curr[idx] = old_val
                no_improve += 1
            else:
                no_improve = 0

    if best_poly is None:
        best_poly = [1] * 12

    if not validate(best_poly):
        best_poly = [1] * 12

    return best_poly
