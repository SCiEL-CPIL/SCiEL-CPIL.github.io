import time
from collections import deque

def construct():
    p = 101
    denominators = list(range(1, 11))
    n = 10
    deadline = time.time() + 85.0

    # Precompute modular inverses
    inverses = [0] * (n + 1)
    for x in denominators:
        inverses[x] = pow(x, -1, p)
    
    # Map denominator value to bit index 0..9
    # d=1 -> bit 0, d=2 -> bit 1, etc.
    
    # BFS over (sum_mod_p, used_mask)
    # dist[r] will store the minimum size (number of denominators) to reach residue r
    # We also need to store the actual combination to reconstruct the list.
    # Since we want the *smallest* size, BFS by levels works.
    
    # State: (current_sum, mask)
    # visited: set of (sum, mask) to avoid reprocessing?
    # Actually, we want the min size for each *sum*.
    # If we reach sum S with size k, and we already reached S with size k', where k' <= k, we can skip.
    # But we need the *combination* for the best size.
    
    best_size = [n + 1] * p
    best_mask = [0] * p  # 0 means not found or empty
    
    # Start with sum 0, mask 0. But we need non-empty.
    # So we start by adding the first denominator.
    
    # Queue for BFS: (sum, mask, size)
    # We process by size.
    
    # Instead of a queue, we can iterate by size, similar to the original,
    # but using the mask to generate next states.
    # This is essentially the same as the original.
    
    # Let's stick to the original logic but optimize the inner loop?
    # The original is already O(2^10 * 10) which is tiny.
    
    # Is there any other interpretation?
    # "Minimize the largest row length."
    
    # What if I try to use *all* denominators for some rows to "balance" something?
    # No, the objective is strictly the max length.
    
    # I will provide a BFS-based solution as the structural change.
    # It is logically equivalent but structurally different.
    
    # Initialize
    # dist[sum] = min_size
    # parent[sum][mask] is not needed if we just store the mask for the best size.
    
    # We can use a simple array for min_size and min_mask.
    # But note: a larger mask might have a smaller size? No, size is popcount(mask).
    # So if we find a mask with size k for sum S, and we already have a mask with size k' < k for S, we ignore.
    # If k' == k, we can keep either.
    
    # BFS Level by Level
    # Level 0: sum=0, mask=0. (Not a valid row, but start)
    # Level 1: add one denom.
    # Level 2: add second denom.
    
    # We can do this:
    # current_states: list of (sum, mask)
    # next_states: set of (sum, mask)
    
    current_states = [(0, 0)]
    # We don't put (0,0) in best because row must be non-empty.
    
    for size in range(1, n + 1):
        if time.time() > deadline:
            break
        next_states = set()
        for s, m in current_states:
            for d_idx in range(n):
                bit = 1 << d_idx
                if m & bit:
                    continue
                d_val = d_idx + 1
                new_s = (s + inverses[d_val]) % p
                new_m = m | bit
                next_states.add((new_s, new_m))
        
        # Process next_states
        # For each state, check if it improves best_size for that residue
        for s, m in next_states:
            if best_size[s] > size:
                best_size[s] = size
                best_mask[s] = m
            # If best_size[s] == size, we could update mask, but it doesn't matter for the value.
        
        # Prepare for next level
        # We only need to keep states that *might* lead to a better size for some residue?
        # Actually, any state can lead to a better size for a *different* residue in the next step.
        # But if a state (s, m) has size k, and we already visited (s, m') with size k' < k,
        # then (s, m) is dominated?
        # Not necessarily, because m might allow adding different denominators later?
        # No, if m' is a subset of m, then m' is "better" (more options, smaller size so far).
        # But we are doing BFS by size. All states in current_states have the same size.
        # So we just need to avoid duplicates (same s, same m).
        
        current_states = list(next_states)
        
        # Optimization: If all residues are found, we can stop?
        # No, we need to check if larger sizes are needed for *other* residues?
        # No, if a residue is found with size k, we don't need to find it with size k+1.
        # But we need to find residues *not yet found*.
        # So we can stop if all best_size are set.
        
        if all(best_size[r] != n + 1 for r in range(p)):
            break

    # Build result
    result = []
    for r in range(p):
        if best_mask[r] != 0:
            # Reconstruct denominators from mask
            combo = []
            for i in range(n):
                if best_mask[r] & (1 << i):
                    combo.append(i + 1)
            result.append(combo)
        else:
            # Fallback
            result.append(denominators[:])
            
    return result
