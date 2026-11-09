import time
import random
import numpy as np

def _find_subset_sum(S, target=200):
    """
    Find a subset of S summing to target.
    Returns the subset as a list, or None.
    Uses DP for efficiency.
    """
    if not S:
        return None
    
    # Map sum to the subset (list of elements)
    # To save memory, we only store the path for the last element added
    # But we need to reconstruct the full subset.
    # For small targets (200) and small sets (100), we can store parent pointers.
    
    n = len(S)
    # dp[s] = (last_index, prev_sum)
    parent = {}
    reachable = {0: None} # sum -> last_index
    
    for i, x in enumerate(S):
        if x > target:
            continue
        # Iterate over a copy of keys to avoid modification during iteration issues, 
        # but we are adding new keys, so we can iterate over existing keys
        new_reachable = {}
        for s, last_idx in list(reachable.items()):
            ns = s + x
            if ns <= target and ns not in reachable:
                new_reachable[ns] = i
        for ns, idx in new_reachable.items():
            if ns not in reachable:
                reachable[ns] = idx
                # Store parent info: to get sum ns, we added x (index idx) to sum ns-x
                # We need to know which index was used for ns-x.
                # reachable[ns-x] gives the last index used for ns-x.
                # But we need the full path.
                # Let's store parent[ns] = (ns-x, idx)
                parent[ns] = (ns - x, idx)
    
    if target not in reachable:
        return None
    
    # Reconstruct
    subset_indices = []
    cur = target
    while cur is not None:
        if cur == 0:
            break
        prev_sum, idx = parent[cur]
        subset_indices.append(idx)
        cur = prev_sum
    
    if not subset_indices:
        return []
        
    return [S[i] for i in subset_indices]

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # Elements are 1 to 100
    all_elements = list(range(1, 101))
    
    # R is the set of removed elements
    R = set()
    
    # We want to minimize |R|
    # Greedy Hitting Set:
    # 1. Find a conflict C in (all_elements \ R)
    # 2. Pick an element from C to add to R
    
    # To pick the best element, we can sample multiple conflicts
    # and count frequencies.
    
    while time.time() - start_time < deadline - 5:
        available = [x for x in all_elements if x not in R]
        
        # Check if available is valid
        conflict = _find_subset_sum(available, 200)
        
        if conflict is None:
            # No conflict found, we are done
            break
            
        # We have a conflict. We need to remove at least one element from it.
        # Which one?
        # Strategy: Sample K conflicts, count frequency of each element in the union of conflicts.
        # Remove the most frequent one.
        
        counts = {}
        sample_size = 50
        found_conflicts = 0
        
        # The first conflict is known
        for x in conflict:
            counts[x] = counts.get(x, 0) + 1
        found_conflicts += 1
        
        # Find more conflicts to get a better frequency estimate
        # To find diverse conflicts, we can randomly permute 'available' 
        # or just run the DP on shuffled subsets?
        # DP is deterministic based on order.
        # Let's shuffle 'available' and run DP to find different conflicts.
        
        for _ in range(sample_size):
            if time.time() - start_time > deadline - 5:
                break
            shuffled = available[:]
            random.shuffle(shuffled)
            c = _find_subset_sum(shuffled, 200)
            if c is not None:
                for x in c:
                    counts[x] = counts.get(x, 0) + 1
                found_conflicts += 1
                
        if found_conflicts == 0:
            # Should not happen since we found one
            break
            
        # Pick the element with highest count
        # Tie-breaker: pick larger element? Smaller?
        # Larger elements are in fewer subsets (pairs/triples) than smaller ones (many combos).
        # Actually, smaller elements are in MORE subsets.
        # If we remove a small element, we break many potential sums.
        # So we should prefer smaller elements?
        # Let's check: 
        # 1 is in {1, 199}, {1, 2, 197}, etc.
        # 100 is in {100, 100} (invalid), {100, 99, 1}, {100, 98, 2}.
        # 1 is in more subsets than 100.
        # So removing 1 is more effective.
        
        # Sort by count (desc), then by value (asc)
        candidates = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        
        if candidates:
            best_elem = candidates[0][0]
            R.add(best_elem)
        else:
            break
            
    # After the loop, verify the result
    final_set = [x for x in all_elements if x not in R]
    
    # Final verification and repair if needed
    # If the greedy approach got stuck or left a conflict, we can try to remove more
    # But the loop condition handles this.
    
    # If we ran out of time, we might have a conflict.
    # We must return a VALID set.
    # So we need a fallback repair.
    
    current_set = final_set
    # Repair loop: while conflict exists, remove an element
    # Use a fast check
    while time.time() - start_time < deadline - 1:
        conflict = _find_subset_sum(current_set, 200)
        if conflict is None:
            break
        # Remove the first element of the conflict
        # Or the smallest?
        elem_to_remove = min(conflict)
        if elem_to_remove in current_set:
            current_set.remove(elem_to_remove)
    
    if not current_set:
        return [100] # Fallback, though 100 alone is valid
    
    return current_set
