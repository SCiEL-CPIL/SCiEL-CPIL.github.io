import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    n = 16
    k = 4
    ALL_MASK = (1 << n) - 1

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    mask_to_idx = {}
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
        mask_to_idx[mask] = len(all_masks) - 1

    m = len(all_masks)  # 1820

    # Precompute list of elements for each mask to speed up intersection/union logic?
    # Or just use bit operations.
    
    # Helper to get conflicts for a pair of masks
    def get_conflicts(a_mask, b_mask):
        inter_mask = a_mask & b_mask
        union_mask = a_mask | b_mask
        rem_mask = ALL_MASK & ~union_mask
        
        inter_list = []
        rem_list = []
        # Unroll loop for speed? n=16 is small.
        for i in range(n):
            bit = 1 << i
            if inter_mask & bit:
                inter_list.append(i)
            elif rem_mask & bit:
                rem_list.append(i)
                
        k_rem = 4 - len(inter_list)
        if k_rem < 0 or k_rem > len(rem_list):
            return []
            
        conflicts = []
        for comb in combinations(rem_list, k_rem):
            c_mask = inter_mask
            for x in comb:
                c_mask |= (1 << x)
            conflicts.append(mask_to_idx[c_mask])
        return conflicts

    in_family = [False] * m
    block_count = [0] * m
    family = []
    
    def add(idx):
        in_family[idx] = True
        family.append(idx)
        a_mask = all_masks[idx]
        # Update block counts for all existing family members
        for t in family[:-1]: # Exclude self
            b_mask = all_masks[t]
            confs = get_conflicts(a_mask, b_mask)
            for c in confs:
                block_count[c] += 1

    def remove(idx):
        in_family[idx] = False
        a_mask = all_masks[idx]
        # Find position in family
        pos = family.index(idx) # O(F)
        # Swap remove
        family[pos] = family[-1]
        family.pop()
        
        # Update block counts
        for t in family:
            b_mask = all_masks[t]
            confs = get_conflicts(a_mask, b_mask)
            for c in confs:
                block_count[c] -= 1

    # 1. Greedy Phase
    # Order: Random or Natural? Random might be better to avoid symmetry traps.
    # But we need to check block_count.
    # Initially all 0.
    # We can just iterate and add if block_count == 0.
    # To make it robust, shuffle the order of candidates.
    
    order = list(range(m))
    random.shuffle(order)
    
    for idx in order:
        if time.time() > deadline:
            break
        if not in_family[idx] and block_count[idx] == 0:
            add(idx)

    best_size = len(family)
    best_family = family[:]
    best_in_family = in_family[:]
    best_block_count = block_count[:]

    # 2. Local Search Phase
    # We store the state to revert if needed, or just manipulate and revert.
    # Since add/remove are symmetric, we can just do them.
    
    # To allow multiple restarts if stuck
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # If we are stuck, we might try a random restart of the whole process?
        # Or just continue local search.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try to add
            # Scan for unblocked
            for idx in range(m):
                if not in_family[idx] and block_count[idx] == 0:
                    add(idx)
                    improved = True
                    break # Restart loop to check for more adds
            
            if not improved:
                # Try swap
                # Pick a random set to remove
                if family:
                    # Try a few random removals
                    attempts = 0
                    max_attempts = 50
                    found_swap = False
                    while attempts < max_attempts and time.time() < deadline - 5:
                        attempts += 1
                        s_old_idx = random.choice(family)
                        remove(s_old_idx)
                        
                        # Check if any set is now unblocked
                        for idx in range(m):
                            if not in_family[idx] and block_count[idx] == 0:
                                add(idx)
                                found_swap = True
                                improved = True
                                break
                                
                        if found_swap:
                            break
                        else:
                            # Revert
                            add(s_old_idx)
                    
                    if not found_swap:
                        # Stuck in this local optimum
                        break

        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            best_in_family = in_family[:]
            best_block_count = block_count[:]
            
        # If stuck, try a new random restart of the greedy phase?
        # Reset state
        # It might be better to just restart the greedy phase with a new seed
        if len(family) == best_size:
             # Optional: Full restart
             pass

    # Restore best state if we messed up? 
    # The code above modifies family in place. 
    # If we break out of the loop, family is the current state.
    # We kept best_family.
    
    # We need to return the best_family found.
    # But we need to ensure it's valid. The algorithm maintains validity.
    
    # Convert best_family to list of lists
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(subset)
        
    return result

# EVOLVE-BLOCK-END
