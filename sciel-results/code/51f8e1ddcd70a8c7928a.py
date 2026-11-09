"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    
    # dp maps sum (Fraction) -> (prev_sum, index)
    # Initial state: sum 0, no previous sum, no index
    dp = {Fraction(0, 1): (None, None)}
    
    for i, d in enumerate(denominators):
        # Collect updates to avoid modifying dictionary during iteration
        updates = {}
        
        for s, (prev_s, prev_idx) in dp.items():
            new_s = s + Fraction(1, d)
            if new_s < 1:
                # Only add if this sum hasn't been reached before
                # Since we process items one by one, if it's not in dp, 
                # it's a new state reachable by including item i.
                if new_s not in dp:
                    updates[new_s] = (s, i)
        
        # Apply updates
        for ns, (ps, pi) in updates.items():
            dp[ns] = (ps, pi)
            
    # Find the maximum sum strictly less than 1
    best_sum = None
    for s in dp.keys():
        if s < 1:
            if best_sum is None or s > best_sum:
                best_sum = s
                
    # Backtrack to reconstruct the indices
    indices = []
    curr_s = best_sum
    while curr_s is not None:
        prev_s, idx = dp[curr_s]
        if idx is not None:
            indices.append(idx)
        curr_s = prev_s
        
    return indices
# EVOLVE-BLOCK-END
