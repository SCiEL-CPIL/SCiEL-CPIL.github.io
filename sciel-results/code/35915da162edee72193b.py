import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    fixed = {1, 2, 5, 11}
    best_set = list(fixed)
    best_size = len(fixed)
    
    # Helper to check if a set is a valid Sidon set
    def is_valid(s):
        sums = set()
        lst = list(s)
        n = len(lst)
        for i in range(n):
            for j in range(i, n):
                s_val = lst[i] + lst[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True

    # Greedy construction
    def greedy_build(initial_set):
        current_set = set(initial_set)
        candidates = [x for x in range(1, 285) if x not in current_set]
        
        # Sort candidates? Smaller numbers are often "harder" to fit later?
        # Or easier? 
        # Let's try sorting by "constraint" or just random order for diversity in restarts.
        # For a deterministic greedy, sort ascending.
        for x in sorted(candidates):
            if time.time() > deadline:
                break
            # Check if x can be added
            # Optimization: Check sums with existing elements
            valid = True
            new_sums = set()
            for y in current_set:
                s_val = x + y
                # We need to know if s_val is already in the set of sums of current_set
                # We don't store current_sums in this function easily without passing it.
                # Let's just call is_valid on the new set? 
                # is_valid is O(N^2). N is small (<20). 
                # So checking is_valid(current_set | {x}) is cheap.
                pass
            
            # Simpler: just try adding and check validity
            # But is_valid is O(N^2). Doing it for every candidate is O(N * N^2) = O(N^3) per step?
            # No, is_valid iterates over all pairs.
            # If we do this for each candidate, it's expensive.
            
            # Better: Maintain a set of forbidden sums?
            # Sums in current_set are fixed.
            # We need x + y not in Sums(current_set) for all y in current_set.
            # And x + x not in Sums(current_set).
            
            # Let's pre-compute sums of current_set
            sums_cs = set()
            cs_list = list(current_set)
            for i in range(len(cs_list)):
                for j in range(i, len(cs_list)):
                    sums_cs.add(cs_list[i] + cs_list[j])
            
            # Check x
            ok = True
            for y in current_set:
                if x + y in sums_cs:
                    ok = False
                    break
            if ok and (x + x in sums_cs):
                ok = False
                
            if ok:
                current_set.add(x)
                
        return current_set

    # Initial greedy
    current_set = greedy_build(fixed)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
        
    # Local Search / Simulated Annealing
    # We will try to improve the size.
    # If we are stuck, we swap elements to find a better configuration.
    
    # To make swaps efficient, we need to quickly check validity.
    # We'll stick to is_valid for simplicity as N is small.
    
    # Random seed for reproducibility in debug, but random is fine
    random.seed(42)
    
    # We use a "plateau search": if we can't add, we swap.
    # We track the best set found.
    
    # To allow escaping local maxima, we might need to accept temporary size reductions?
    # Or just swap within same size.
    # Given the gap between greedy and optimal, swaps within same size are likely sufficient to find a "hole" for a new element.
    
    while time.time() < deadline:
        # Try to add
        added = False
        candidates = [x for x in range(1, 285) if x not in current_set]
        random.shuffle(candidates) # Random order to avoid bias
        
        for x in candidates:
            # Check if x can be added to current_set
            # Efficient check:
            sums_cs = set()
            cs_list = list(current_set)
            for i in range(len(cs_list)):
                for j in range(i, len(cs_list)):
                    sums_cs.add(cs_list[i] + cs_list[j])
            
            ok = True
            for y in current_set:
                if x + y in sums_cs:
                    ok = False
                    break
            if ok and (x + x in sums_cs):
                ok = False
                
            if ok:
                current_set.add(x)
                added = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                # If we reached 20, we can return early? 
                # The max is 20. If we hit 20, we are optimal.
                if best_size == 20:
                    return best_set
                break # Restart candidate search with new set
                
        if not added:
            # Local maximum. Try to swap a non-fixed element with a candidate.
            # Pick a random element to remove (not fixed)
            removable = [x for x in current_set if x not in fixed]
            if not removable:
                break # Can't improve
            
            y_to_remove = random.choice(removable)
            
            # Pick a random candidate to add
            candidates = [x for x in range(1, 285) if x not in current_set]
            if not candidates:
                break
            x_to_add = random.choice(candidates)
            
            # Try the swap
            new_set = (current_set - {y_to_remove}) | {x_to_add}
            
            # Check validity
            if is_valid(new_set):
                # Accept the swap (same size, different config)
                current_set = new_set
                # Note: We don't update best_size here as size is same.
                # But this new config might allow an addition in the next loop iteration.
            else:
                # If swap invalid, we could try other swaps, but for now just loop again
                # To avoid infinite loop on same swap, we rely on random choice changing.
                pass

    return best_set
