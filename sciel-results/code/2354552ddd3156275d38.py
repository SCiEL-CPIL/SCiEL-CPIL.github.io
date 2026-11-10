"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit (5s buffer)
    
    n = 256
    nums = list(range(1, n + 1))
    
    # Build conflict masks
    # mask[i] is a bitset where bit j is 1 if i and j conflict
    # We use 0-indexed for bits: bit 0 corresponds to number 1, bit 1 corresponds to number 2, etc.
    # So number x corresponds to bit (x-1)
    conflict_masks = [0] * (n + 1)
    degrees = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                # i conflicts with j
                conflict_masks[i] |= (1 << (j - 1))
                conflict_masks[j] |= (1 << (i - 1))
                degrees[i] += 1
                degrees[j] += 1
                
    # Greedy Initialization
    # Sort by degree ascending (Minimum Degree Heuristic)
    # This often yields a larger initial independent set
    sorted_nums = sorted(nums, key=lambda x: degrees[x])
    
    S_mask = 0
    S_list = []
    for i in sorted_nums:
        if time.time() > deadline:
            break
        # Check if i conflicts with any in S
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
            
        # Iterate through nodes in S to remove
        # Randomizing the order might help escape local minima, but deterministic is fine for now
        # We can also iterate in reverse or random order. Let's stick to sorted by degree or just list order.
        for u in S_list:
            if time.time() > deadline:
                return sorted(best_set)
            
            # Candidates compatible with S \ {u}
            S_mask_no_u = S_mask & ~(1 << (u - 1))
            
            # Find all v not in S such that v is compatible with S_mask_no_u
            # v is not in S: ~(S_mask)
            # v is compatible: ~(conflict_masks[v])
            # We can't easily do a bitset intersection for "v not in S AND compatible with S_mask_no_u" 
            # because compatibility depends on v.
            # So we iterate v from 1 to n.
            
            candidates_mask = 0
            for v in range(1, n + 1):
                if (S_mask >> (v - 1)) & 1: 
                    continue # v is in S
                if (conflict_masks[v] & S_mask_no_u) == 0:
                    candidates_mask |= (1 << (v - 1))
            
            # If we have at least 2 candidates, check for a non-conflicting pair
            if candidates_mask:
                # Get list of candidates for iteration
                # We can just iterate through bits set in candidates_mask
                # Or iterate 1..n and check if bit is set.
                
                found_pair = False
                # Iterate through candidates v
                # To be efficient, we can extract indices from candidates_mask
                # But 1..256 loop is fast enough in Python if optimized? 
                # Let's use a while loop on the mask to iterate set bits.
                
                cm = candidates_mask
                while cm and not found_pair:
                    # Get lowest set bit
                    v_bit = cm & -cm
                    v_idx = v_bit.bit_length() - 1
                    v = v_idx + 1
                    cm ^= v_bit
                    
                    # Check if v has a non-neighbor in candidates_mask
                    # Non-neighbors of v in C:
                    # C_mask & ~(conflict_masks[v] | v_bit)
                    # We exclude v itself from its own non-neighbor check
                    potential_w_mask = candidates_mask & ~(conflict_masks[v] | v_bit)
                    
                    if potential_w_mask:
                        # Found w
                        w_bit = potential_w_mask & -potential_w_mask
                        w_idx = w_bit.bit_length() - 1
                        w = w_idx + 1
                        
                        # Perform swap
                        best_set.remove(u)
                        best_set.add(v)
                        best_set.add(w)
                        best_size += 1
                        improved = True
                        found_pair = True
                        break
                
                if found_pair:
                    # Restart the outer loop with the new set
                    break
                
    return sorted(best_set)
# EVOLVE-BLOCK-END
