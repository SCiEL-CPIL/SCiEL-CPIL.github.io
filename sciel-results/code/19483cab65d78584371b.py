import time
import random

def construct():
    """
    Construct a set of distinct integers in 1..256 such that subset sums of
    different cardinalities never collide.
    
    Strategy: 
    - Greedy phase to build a large base set quickly using cardinality-indexed
      subset-sum DP.
    - Multiple randomized restarts with different orderings to escape local optima.
    - Within a small time budget, try local-search-style swaps: replace an element
      with a different one to see if the set size can grow.
    
    The key invariant: for each cardinality k, T[k] is the set of all subset sums
    of size k. We require T[k] and T[j] disjoint for k != j.
    """
    deadline = time.time() + 55.0
    
    def is_valid_addition(T, current_size, x):
        """Check if adding x maintains the disjointness invariant."""
        max_k = current_size + 1
        for k in range(1, max_k + 1):
            if not T[k-1]:
                continue
            new_sums = {s + x for s in T[k-1]}
            for j in range(0, current_size + 1):
                if j == k:
                    continue
                if T[j] and (new_sums & T[j]):
                    return False
        return True
    
    def add_element(T, x):
        """Add x to T in-place, returning old T for rollback."""
        old_T = [s.copy() for s in T]
        new_max_k = len(old_T) - 1  # len(T) is 61, so max k is 60
        # Actually T has 61 entries, indices 0..60
        for k in range(1, 61):
            if not old_T[k-1]:
                break
            for s in old_T[k-1]:
                T[k].add(s + x)
        return old_T
    
    def greedy_construct(order):
        """Build a set using the given order of candidates."""
        T = [set() for _ in range(61)]
        T[0] = {0}
        current_set = []
        used = set()
        
        for x in order:
            if time.time() > deadline:
                break
            if len(current_set) >= 60:
                break
            if x in used:
                continue
            if is_valid_addition(T, len(current_set), x):
                old_T = add_element(T, x)
                current_set.append(x)
                used.add(x)
        
        return current_set, T, used
    
    # Phase 1: Try several different orderings
    best_set = []
    best_T = None
    best_used = None
    
    # Ordering 1: ascending
    order1 = list(range(1, 257))
    # Ordering 2: descending
    order2 = list(range(256, 0, -1))
    # Ordering 3: random
    order3 = list(range(1, 257))
    random.shuffle(order3)
    # Ordering 4: alternating small/large
    smalls = list(range(1, 129))
    larges = list(range(256, 128, -1))
    order4 = []
    si, li = 0, 0
    while si < len(smalls) or li < len(larges):
        if si < len(smalls):
            order4.append(smalls[si])
            si += 1
        if li < len(larges):
            order4.append(larges[li])
            li += 1
    
    for order in [order1, order2, order3, order4]:
        if time.time() > deadline:
            break
        s, T, used = greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s
            best_T = T
            best_used = used
    
    # Phase 2: Local search - try swapping elements to allow adding more
    # If we can't add any more, try removing one element and see if we can
    # add two or more in its place
    
    if len(best_set) < 60 and best_T is not None:
        current_set = best_set[:]
        T = [s.copy() for s in best_T]
        used = best_used.copy()
        
        # Try to find any element we can add
        def find_addable(current_set, T, used):
            for x in range(1, 257):
                if x in used:
                    continue
                if is_valid_addition(T, len(current_set), x):
                    return x
            return None
        
        # Keep adding if possible
        while len(current_set) < 60:
            if time.time() > deadline:
                break
            x = find_addable(current_set, T, used)
            if x is None:
                break
            old_T = add_element(T, x)
            current_set.append(x)
            used.add(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            best_T = T
            best_used = used
    
    # Phase 3: If still not at 60, try removing one element and rebuilding
    if len(best_set) < 60:
        for remove_idx in range(len(best_set)):
            if time.time() > deadline:
                break
            
            # Remove element
            removed = best_set[remove_idx]
            # Rebuild T from scratch without this element
            base_set = [x for i, x in enumerate(best_set) if i != remove_idx]
            T_new = [set() for _ in range(61)]
            T_new[0] = {0}
            used_new = set(base_set)
            
            for x in base_set:
                old_T = [s.copy() for s in T_new]
                for k in range(1, len(base_set) + 2):
                    if not old_T[k-1]:
                        break
                    for s in old_T[k-1]:
                        T_new[k].add(s + x)
            
            # Now try greedy addition from here
            order = list(range(1, 257))
            random.shuffle(order)
            s, T, used = greedy_construct(order)
            # Merge: start from base_set, then add
            # Actually let's just do greedy from scratch with the removed element excluded first
            # This is expensive; only do it if we have time
            
            if len(s) > len(best_set):
                best_set = s
                best_T = T
                best_used = used
    
    # Ensure valid return
    if not best_set:
        # Fallback: just return [1]
        best_set = [1]
    
    return [int(x) for x in best_set]
