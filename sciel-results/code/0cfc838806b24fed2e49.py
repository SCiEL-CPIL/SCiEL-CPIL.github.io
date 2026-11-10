"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = 256
    nums = list(range(1, n + 1))
    
    # Build conflict masks
    # mask[i] is a bitset where bit j is 1 if i and j conflict
    # We use 0-indexed for bits: bit 0 corresponds to number 1, bit 1 to number 2, etc.
    # So number x corresponds to bit (x-1)
    conflict_masks = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                # i conflicts with j
                conflict_masks[i] |= (1 << (j - 1))
                conflict_masks[j] |= (1 << (i - 1))
                
    # Greedy Initialization
    # Start with empty set, add nodes in order 1..256 if compatible
    S_mask = 0
    S_list = []
    for i in range(1, n + 1):
        if time.time() > deadline:
            break
        # Check if i conflicts with any in S
        # S_mask has bits set for numbers in S
        # conflict_masks[i] has bits set for numbers conflicting with i
        # Intersection should be 0
        if (conflict_masks[i] & S_mask) == 0:
            S_mask |= (1 << (i - 1))
            S_list.append(i)
            
    best_set = set(S_list)
    best_size = len(best_set)
    
    # Local Search: 1-for-2 swaps
    # Try to remove u from S, and add two vertices v, w not in S
    # such that v, w are compatible with S\{u} and with each other.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(best_set)
        S_mask = 0
        for x in S_list:
            S_mask |= (1 << (x - 1))
            
        for u in S_list:
            if time.time() > deadline:
                return sorted(best_set)
            
            # Candidates compatible with S \ {u}
            # S_mask without u
            S_mask_no_u = S_mask & ~(1 << (u - 1))
            
            # Find all v not in S such that v is compatible with S_mask_no_u
            candidates = []
            for v in range(1, n + 1):
                if (S_mask >> (v - 1)) & 1: 
                    continue # v is in S
                if (conflict_masks[v] & S_mask_no_u) == 0:
                    candidates.append(v)
            
            # Try to find two candidates v, w that do not conflict with each other
            found_pair = False
            if len(candidates) >= 2:
                # Optimization: if candidates is large, checking all pairs is O(K^2)
                # But K is usually small in a maximal independent set context?
                # Actually, if S is maximal, candidates are vertices that could be added if we remove u.
                # Let's just check pairs.
                for i in range(len(candidates)):
                    v = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        w = candidates[j]
                        # Check if v and w conflict
                        if (conflict_masks[v] & (1 << (w - 1))) == 0:
                            # Swap: remove u, add v, w
                            best_set.remove(u)
                            best_set.add(v)
                            best_set.add(w)
                            best_size += 1
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
            
            if found_pair:
                # Restart the outer loop with the new set
                break
                
    return sorted(best_set)
# EVOLVE-BLOCK-END
