"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np

def construct():
    N = 512
    nums = list(range(1, N))
    
    # Weights: 1 / (N - a)
    # We want to maximize sum of weights for selected numbers.
    # Sort indices by weight descending (which means a descending from N-1 to 1)
    # a = N-1 has weight 1/(1)
    # a = N-2 has weight 1/(2)
    # So sorting by a descending is sorting by weight descending.
    
    # Step 1: Greedy Initialization
    selected = []
    # Iterate from largest a to smallest a
    for a in range(N - 1, 0, -1):
        is_coprime = True
        for s in selected:
            if math.gcd(a, s) != 1:
                is_coprime = False
                break
        if is_coprime:
            selected.append(a)
            
    # Convert to set for faster lookup if needed, but list is fine for small N
    selected_set = set(selected)
    
    # Calculate initial score
    def calc_score(s_list):
        return sum(1.0 / (N - a) for a in s_list)
    
    current_score = calc_score(selected)
    
    # Step 2: Local Search (Hill Climbing)
    # Try to swap out a selected element s with an unselected element u
    # if the new set is valid and score increases.
    
    # Precompute weights for faster access
    weights = {a: 1.0 / (N - a) for a in nums}
    
    improved = True
    # Limit iterations to prevent infinite loops or excessive time, though convergence should be fast
    max_iters = 100
    
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        
        # Create a list of unselected numbers
        unselected = [a for a in nums if a not in selected_set]
        
        # Try all pairs (s in selected, u in unselected)
        # To be efficient, we can check validity first.
        # Validity of (selected - {s}) U {u}:
        # 1. gcd(u, t) == 1 for all t in (selected - {s})
        
        for s in selected[:]: # Iterate over a copy because we might modify
            # Removing s, we need to check if u can be added to the rest
            # Let rest = selected - {s}
            # If u is coprime with all in rest, then the new set is valid.
            
            # Optimization: If weight[u] <= weight[s], swapping won't improve score (since we remove s and add u, net change is w[u]-w[s]).
            # Actually, we remove s and add u. Score change = w[u] - w[s].
            # So we only care if w[u] > w[s].
            
            for u in unselected:
                if weights[u] <= weights[s]:
                    continue
                
                # Check if u is coprime with all other selected elements
                valid = True
                for t in selected:
                    if t == s:
                        continue
                    if math.gcd(u, t) != 1:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    selected.remove(s)
                    selected.append(u)
                    selected_set.remove(s)
                    selected_set.add(u)
                    
                    # Update score
                    current_score += weights[u] - weights[s]
                    improved = True
                    # Restart inner loop? Or just continue? 
                    # Since the set changed, the 'unselected' list is stale and 'selected' changed.
                    # It's safer to break and restart the while loop to re-evaluate all possibilities
                    # with the new state.
                    break
            
            if improved:
                break # Break out of s loop, restart while loop
        
    return selected

# EVOLVE-BLOCK-END
