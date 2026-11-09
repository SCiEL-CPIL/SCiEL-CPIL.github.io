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
    
    n = len(S)
    # dp[s] = (last_index, prev_sum)
    parent = {}
    reachable = {0: None} # sum -> last_index
    
    for i, x in enumerate(S):
        if x > target:
            continue
        # Iterate over a copy of keys to avoid modification during iteration issues
        new_reachable = {}
        for s, last_idx in list(reachable.items()):
            ns = s + x
            if ns <= target and ns not in reachable:
                new_reachable[ns] = i
        for ns, idx in new_reachable.items():
            if ns not in reachable:
                reachable[ns] = idx
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
    
    # Phase 1: Greedy Removal
    while time.time() - start_time < deadline - 20: # Leave time for addition
        available = [x for x in all_elements if x not in R]
        
        # Check if available is valid
        conflict = _find_subset_sum(available, 200)
        
        if conflict is None:
            # No conflict found, we are done with removal
            break
            
        # We have a conflict. We need to remove at least one element from it.
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
        for _ in range(sample_size):
            if time.time() - start_time > deadline - 20:
                break
            shuffled = available[:]
            random.shuffle(shuffled)
            c = _find_subset_sum(shuffled, 200)
            if c is not None:
                for x in c:
                    counts[x] = counts.get(x, 0) + 1
                found_conflicts += 1
                
        if found_conflicts == 0:
            break
            
        # Pick the element with highest count
        candidates = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        
        if candidates:
            best_elem = candidates[0][0]
            R.add(best_elem)
        else:
            break
            
    # Current set
    current_set = set(all_elements) - R
    
    # Phase 2: Greedy Addition
    # Try to add back removed elements
    # Order: Smallest first? Largest first?
    # Smallest elements are more "versatile" but also more likely to cause conflicts.
    # Let's try adding in random order or sorted order.
    # Sorted ascending might be good because small elements complete many sums.
    # If we can add a small element, it's a big win.
    
    removed_list = sorted(list(R))
    
    # We can iterate multiple times until no more additions
    improved = True
    while improved and time.time() - start_time < deadline - 2:
        improved = False
        for x in removed_list:
            if time.time() - start_time > deadline - 2:
                break
            if x in current_set:
                continue
            
            # Check if adding x creates a conflict
            # New set: current_set U {x}
            # We only need to check conflicts involving x?
            # No, adding x might create a conflict that didn't exist before?
            # No, if current_set is valid, any new conflict MUST involve x.
            # So we just need to check if there is a subset of (current_set U {x}) summing to 200 that contains x.
            # This is equivalent to checking if there is a subset of current_set summing to 200 - x.
            
            target_remainder = 200 - x
            if target_remainder < 0:
                # x > 200, cannot be part of sum 200 with positive integers
                # But x <= 100, so this won't happen.
                current_set.add(x)
                R.remove(x)
                improved = True
                continue
                
            # Check if current_set has a subset summing to target_remainder
            # We can use the same DP, but on current_set.
            # To save time, we can write a specialized check.
            
            # Quick check: if target_remainder == 0, then x itself sums to 200? No, x<200.
            # If target_remainder is in current_set? Then {x, y} sums to 200.
            # We need a general subset sum check.
            
            # Reuse _find_subset_sum?
            # It returns the subset. We just need to know if it exists.
            # But _find_subset_sum is O(N*Target).
            # N=100, Target=200. Fast.
            
            # However, _find_subset_sum reconstructs the subset. We don't need that.
            # Let's write a fast existence check.
            
            def _has_subset_sum(S, target):
                if target < 0:
                    return False
                if target == 0:
                    return True
                bits = 1
                for x in S:
                    if x <= target:
                        bits |= (bits << x)
                return bool(bits & (1 << target))
            
            if not _has_subset_sum(current_set, target_remainder):
                # Safe to add
                current_set.add(x)
                R.remove(x)
                improved = True
            else:
                # Conflict exists, cannot add
                pass
    
    # Final check
    if not current_set:
        return [100]
        
    return list(current_set)
