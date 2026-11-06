import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    
    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1
    
    # Start with upper half
    A = set(range(64, 129))
    ind = np.zeros(N_max, dtype=np.int32)
    for x in A:
        ind[x] = 1
        
    # Compute initial counts
    conv = np.convolve(ind, ind, mode='full')
    counts = np.zeros(R, dtype=np.int32)
    for i, s in enumerate(range(T_min, T_max + 1)):
        c_ord = conv[s]
        if s % 2 == 0:
            half = s // 2
            n_eq = 1 if (half < N_max and ind[half] == 1) else 0
        else:
            n_eq = 0
        counts[i] = (c_ord + n_eq) // 2
        
    max_count = int(np.max(counts))
    best_max = max_count
    best_A = set(A)
    
    # Precompute complement list for faster y selection?
    # Or just use a set for A for O(1) lookup.
    
    # Function to calculate delta max for a swap x->y
    # We can simulate the count changes on a copy of counts?
    # No, we can just compute the affected sums.
    
    def get_affected_sums(x):
        sums = []
        for a in A:
            s = a + x
            if T_min <= s <= T_max:
                sums.append(s)
        return sums

    # Greedy step
    # Pick x, then find best y.
    
    no_improve = 0
    current_max = max_count
    
    while time.time() < deadline:
        if len(A) == 0: break
        
        # Pick x
        # Bias towards x that is in high count sums?
        # Just random for now
        x = random.choice(list(A))
        
        best_y = None
        best_new_max = float('inf')
        best_sq_sum = float('inf')
        
        # Evaluate all y not in A
        # To speed up, we can sample if N_max is large, but 129 is small.
        for y in range(N_max):
            if y in A:
                continue
                
            # Simulate swap x -> y
            # We need to know the new counts for affected sums.
            # Affected sums are those involving x or y.
            
            # Let's just do the full O(|A|) update on a temporary array?
            # Or just calculate the change for the specific sums.
            
            # Sums affected by removing x:
            # For each a in A, s = a+x.
            # Sums affected by adding y:
            # For each a in A (without x, with y), s = a+y.
            
            # This is tricky to do without modifying global state.
            # Let's just use the try_swap mechanism but "dry run" it?
            # Or just accept the cost of O(|A|) for each y candidate.
            # 129 candidates * 60 ops = 7740 ops.
            # 10,000 iterations * 7740 = 77M ops.
            # Python is slow. 77M ops might take ~5-10s.
            # So we can do ~10-20k iterations.
            
            # Let's optimize the inner loop.
            # Instead of iterating A for each y, we can precompute the "profile" of x and y?
            
            # Profile of z: dict {sum: count_change}
            # For x: -1 for each a+x
            # For y: +1 for each a+y (where a in A\{x})
            
            # Let's just implement the dry-run update.
            
            # Create a delta array for counts
            # We only care about sums in [128, 256]
            
            # 1. Subtract x contributions
            # 2. Add y contributions
            
            # To avoid allocating a new array, we can just track the changes to `counts`
            # and revert them.
            
            # But we need to compare max.
            # We can just update `counts` in place, find max, then revert.
            
            # This is safe as long as we revert.
            
            # Save old counts for affected sums?
            # Or just update and revert.
            
            # Let's do update and revert.
            
            # 1. Remove x
            for a in A:
                s = a + x
                if T_min <= s <= T_max:
                    counts[s - T_min] -= 1
            # A.remove(x) implicitly assumed for the next step's logic?
            # The "Add y" step iterates A. If x is still in A, we add x+y.
            # But in the new set, x is gone.
            # So the "Add y" step should iterate A \ {x}.
            
            # So:
            # 1. Decrement counts for a+x for all a in A.
            # 2. Increment counts for a+y for all a in A \ {x}.
            #    Note: if a=y, s=2y.
            
            # Let's implement this carefully.
            
            # Step 1: Remove x
            # We don't modify A yet, just counts.
            for a in A:
                s = a + x
                if T_min <= s <= T_max:
                    counts[s - T_min] -= 1
            
            # Step 2: Add y
            # Iterate a in A. If a == x, skip (since x is removed).
            for a in A:
                if a == x:
                    continue
                s = a + y
                if T_min <= s <= T_max:
                    counts[s - T_min] += 1
                # Also need to handle y+y?
                # The loop `for a in A` includes y? No, y is not in A yet.
                # So we need to add the (y,y) pair separately if 2y is in range.
                # Wait, if a=y, s=2y. But y is not in A.
                # So we must add 1 to counts[2y-T_min] if 2y is in range.
            if T_min <= 2*y <= T_max:
                counts[2*y - T_min] += 1
                
            # Now counts reflects the new set A' = A \ {x} U {y}
            
            # Check validity
            if np.any(counts < 0) or np.any(counts == 0):
                # Revert
                # Reverse the operations
                if T_min <= 2*y <= T_max:
                    counts[2*y - T_min] -= 1
                for a in A:
                    if a == x:
                        continue
                    s = a + y
                    if T_min <= s <= T_max:
                        counts[s - T_min] -= 1
                for a in A:
                    s = a + x
                    if T_min <= s <= T_max:
                        counts[s - T_min] += 1
                continue # Invalid, try next y
            
            new_max = int(np.max(counts))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_y = y
                # Optional: calculate sum of squares for tie-breaking
                # best_sq_sum = np.sum(counts*counts)
                
            # Revert to try next y
            if T_min <= 2*y <= T_max:
                counts[2*y - T_min] -= 1
            for a in A:
                if a == x:
                    continue
                s = a + y
                if T_min <= s <= T_max:
                    counts[s - T_min] -= 1
            for a in A:
                s = a + x
                if T_min <= s <= T_max:
                    counts[s - T_min] += 1
        
        if best_y is None:
            break
            
        # Apply the best swap
        y = best_y
        # We need to actually apply it to A and ind
        # We can just use the same logic as above but without reverting
        for a in A:
            s = a + x
            if T_min <= s <= T_max:
                counts[s - T_min] -= 1
        A.remove(x)
        ind[x] = 0
        
        for a in A:
            if a == x: # x is already removed, so this check is redundant but safe
                continue
            s = a + y
            if T_min <= s <= T_max:
                counts[s - T_min] += 1
        if T_min <= 2*y <= T_max:
            counts[2*y - T_min] += 1
        A.add(y)
        ind[y] = 1
        
        current_max = best_new_max
        
        if current_max < best_max:
            best_max = current_max
            best_A = set(A)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 1000:
                # Random restart or perturbation?
                # Let's just break and try a different x selection strategy?
                # For now, just continue, might find a way out.
                # To avoid getting stuck, we can accept equal max with small prob.
                # But the greedy search always picks the best y.
                # If the best y doesn't improve max, we are stuck.
                # We should allow non-improving moves.
                pass

    return list(best_A)
