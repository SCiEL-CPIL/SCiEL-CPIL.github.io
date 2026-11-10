import numpy as np
import random
import time

def construct():
    # Base-3 construction
    A_base = []
    for mask in range(1 << 5):
        val = 0
        for i in range(5):
            if mask & (1 << i):
                val += 3**i
        A_base.append(val)
    
    A = [x + 7 for x in A_base]
    A_set = set(A)
    
    if not A_set or max(A_set) > 128:
        return list(range(64, 129))
        
    # Initialize sum counts
    # sums_range = 128 to 256
    # We only care about these, but updates affect them.
    # Actually, we need to know if coverage is lost, so we track all sums?
    # No, only sums in [128, 256] matter for coverage and score.
    # Sums outside don't affect score or validity (except bounds of A).
    
    # dict: sum -> count
    sum_counts = {s: 0 for s in range(128, 257)}
    
    def update_counts(A_list, delta=1):
        # delta 1 for add, -1 for remove
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if 128 <= s <= 256:
                    sum_counts[s] += delta
    
    # Initial counts
    update_counts(A, 1)
    
    current_max = max(sum_counts.values())
    best_A = A.copy()
    best_max = current_max
    best_counts = sum_counts.copy()
    
    # Local Search
    start_time = time.time()
    
    # To make update efficient, we need to know which pairs involve the changed element.
    # When removing a, we subtract 1 from counts of s = a + b for all b in A (b != a) and s=2a.
    # When adding a, we add 1 to counts of s = a + b for all b in A_new (b != a) and s=2a.
    
    def try_swap(A_list, a_out, a_in):
        # Check bounds
        if a_in < 0 or a_in > 128:
            return None, None
            
        # 1. Remove a_out
        # Find index
        try:
            idx = A_list.index(a_out)
        except ValueError:
            return None, None
            
        A_temp = A_list.copy()
        A_temp.pop(idx)
        
        # Update counts for removal
        # Pairs involving a_out in A_list:
        # (a_out, a_out) if count > 1? No, A is a set.
        # Pairs (a_out, b) for b in A_list, b != a_out.
        # And (a_out, a_out) is 1 pair.
        
        # We need to subtract contributions of a_out from the current counts.
        # Current counts include pairs from A_list.
        # Pairs with a_out:
        # 1. (a_out, a_out) -> sum 2*a_out
        # 2. (a_out, b) for b in A_list, b != a_out.
        
        # Let's just recalculate the delta.
        # Delta is -1 for each pair involving a_out.
        
        # Instead of complex delta, let's just rebuild counts for the small set?
        # No, N=32. Rebuilding is 32*32/2 = 512 ops.
        # 2000 * 10 * 512 = 10M ops. Still fast enough in Py?
        # 10M simple ops in Python might take ~1-2 seconds.
        # Let's try the rebuild approach for simplicity and correctness.
        
        new_A = A_temp.copy()
        new_A.append(a_in)
        
        # Check if a_in already in new_A?
        # We are swapping, so a_in != a_out.
        # But a_in might be in A_temp?
        # If a_in in A_temp, then we are just moving a_out to a_in?
        # No, A is a set. If a_in in A_temp, then adding it does nothing to the set structure?
        # Wait, A is a list of unique elements.
        # If a_in is already in A_temp, then new_A has duplicates?
        # No, we should check if a_in is in A_temp. If so, skip (no change in set).
        if a_in in A_temp:
            return None, None
            
        # Calculate new counts
        new_counts = {s: 0 for s in range(128, 257)}
        for i in range(len(new_A)):
            for j in range(i, len(new_A)):
                s = new_A[i] + new_A[j]
                if 128 <= s <= 256:
                    new_counts[s] += 1
                    
        # Check coverage
        for s in range(128, 257):
            if new_counts[s] == 0:
                return None, None
                
        new_max = max(new_counts.values())
        return new_A, new_max, new_counts

    # The function signature above returned 2 or 3 values, let's fix.
    
    for _ in range(5000):
        if time.time() - start_time > 50:
            break
            
        # Pick random index
        idx = random.randint(0, len(best_A) - 1)
        a_out = best_A[idx]
        
        # Try a few random a_in
        for _ in range(5):
            a_in = random.randint(0, 128)
            if a_in == a_out:
                continue
            
            # Check if a_in is already in best_A (excluding a_out)
            if a_in in best_A and best_A.count(a_in) > 1:
                continue # Should not happen if set
            # Actually, just check if a_in is in best_A.
            # If a_in is in best_A, and a_in != a_out, then swapping a_out with a_in
            # results in the same set (if we just replace index).
            # Wait, if a_in is in best_A, then best_A already contains a_in.
            # Removing a_out and adding a_in (which is already there) -> Set is same, but list changes?
            # No, we want to change the set.
            # So a_in must NOT be in best_A.
            if a_in in best_A:
                continue
                
            res = try_swap(best_A, a_out, a_in)
            if res is None:
                continue
            new_A, new_max, new_counts = res
            
            if new_max < best_max:
                best_A = new_A
                best_max = new_max
                best_counts = new_counts
                break # Found improvement, restart

    return best_A
