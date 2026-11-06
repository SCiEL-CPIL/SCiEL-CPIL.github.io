"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction

    deadline = time.time() + 55  # 55 second deadline for safety

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # Precompute suffix sums of reciprocals
    # suffix[i] is the sum of 1/denominators[j] for j from i to n-1
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + Fraction(1, denominators[i])

    # DP state: set of reachable sums
    reachable = {Fraction(0)}
    # Path reconstruction: map sum -> (prev_sum, index)
    path = {Fraction(0): None}
    
    best = Fraction(0)

    for i in range(n):
        if time.time() > deadline:
            break
            
        d = denominators[i]
        v = Fraction(1, d)
        
        # Generate new sums
        new_sums = set()
        for s in reachable:
            ns = s + v
            if ns < 1:
                new_sums.add(ns)
                if ns not in path:
                    path[ns] = (s, i)
        
        reachable = reachable.union(new_sums)
        
        # Update best
        # Find the max sum in reachable that is < 1
        # Since all sums in reachable are < 1 by construction (we filter ns < 1, and 0 < 1)
        # We can just take the max of the set
        current_best = max(reachable)
        if current_best > best:
            best = current_best
            
        # Prune
        # If s + suffix[i+1] <= best, then s cannot lead to a better solution
        rem_sum = suffix[i + 1]
        # We must keep sums that are > best - rem_sum
        # i.e., s > best - rem_sum
        # Or s + rem_sum > best
        reachable = {s for s in reachable if s + rem_sum > best}
        
        # Optional: Clean up path for pruned sums to save memory?
        # Not strictly necessary if memory is fine, but good practice.
        # However, iterating over path keys to delete is O(size).
        # Given small size, we can skip explicit deletion or do it if size is large.
        # Let's just let it be. The final trace will only follow one path.

    # Recover indices from best
    # If best is 0, return [0] (index for 1/2) as fallback? 
    # But best should be > 0 because 1/2 < 1.
    
    target_sum = best
    indices = []
    current = target_sum
    while path[current] is not None:
        prev_sum, idx = path[current]
        indices.append(idx)
        current = prev_sum
        
    indices.sort()
    
    # Fallback if something went wrong (e.g. empty list)
    if not indices:
        indices = [0]
        
    return indices
# EVOLVE-BLOCK-END
