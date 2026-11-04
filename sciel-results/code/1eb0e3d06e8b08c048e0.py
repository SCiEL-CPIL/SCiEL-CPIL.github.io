import math
import random

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Search over base-B binary expansion full sets and subsets via hill climbing
    for B in range(2, 20):
        for n in range(2, 16):
            # Base elements are B^0 ... B^(n-1)
            # U is a subset of {0} U {B^0, ..., B^(n-1)}
            # We represent U by a bitmask of length n for the non-zero elements
            # 0 is always included
            
            # Initialize candidates
            # 1. Full set
            # 2. Random sets
            # 3. Sparse sets
            
            candidates = []
            
            # Full set
            candidates.append((1 << n) - 1)
            
            # Random sets
            for _ in range(5):
                mask = 0
                for j in range(n):
                    if random.random() < 0.5:
                        mask |= (1 << j)
                candidates.append(mask)
                
            # Sparse sets (powers of B only)
            candidates.append(0) # Only 0? No, size 1, score 0. 
            # Just one element
            candidates.append(1)
            candidates.append(1 << (n-1))

            # Local search (Hill Climbing)
            for start_mask in candidates:
                current_mask = start_mask
                # Ensure at least one non-zero element? 
                # If mask is 0, U=[0], score 0.
                # We want to maximize score.
                
                improved = True
                while improved:
                    improved = False
                    # Calculate current score
                    # Build U from mask
                    U_curr = [0]
                    for j in range(n):
                        if current_mask & (1 << j):
                            U_curr.append(B ** j)
                    current_score = calc_score(U_curr)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best = U_curr
                        
                    # Try flipping each bit
                    for j in range(n):
                        new_mask = current_mask ^ (1 << j)
                        # Avoid empty set (mask 0) if it leads to score 0, 
                        # but calc_score handles it.
                        U_new = [0]
                        for k in range(n):
                            if new_mask & (1 << k):
                                U_new.append(B ** k)
                        new_score = calc_score(U_new)
                        
                        if new_score > current_score:
                            current_mask = new_mask
                            improved = True
                            break # Restart loop with new best

    return best
