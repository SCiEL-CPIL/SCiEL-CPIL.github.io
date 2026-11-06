import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    n = 4096
    max_sum = 3 * n
    
    # Use a boolean array for O(1) lookup and easy vectorized updates
    # Note: Python sets are also fast, but numpy boolean indexing is very fast for batches.
    # However, updating a numpy array with a list of indices is `sums_mask[indices] = True`.
    # Checking collision: `np.any(sums_mask[indices])`.
    
    best_set = []
    best_size = 0
    
    def try_fill(skeleton, order_func, label=""):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        # Initialize state
        S_list = sorted(skeleton)
        # Validate skeleton first (should be valid by construction, but just in case)
        if not S_list:
            S = np.array([], dtype=np.int64)
        else:
            S = np.array(S_list, dtype=np.int64)
        sums_mask = np.zeros(max_sum + 1, dtype=bool)
        
        # Add skeleton sums
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for k_idx in range(j, k):
                    s_val = S[i] + S[j] + S[k_idx]
                    if sums_mask[s_val]:
                        return # Skeleton invalid
                    sums_mask[s_val] = True
        
        # Determine candidate order
        # We want to try to add numbers not in S
        # Using a set for lookup is fast for 4096 elements
        S_set = set(S_list)
        candidates = [x for x in range(1, n + 1) if x not in S_set]
        candidates = order_func(candidates)
        
        current_S = list(S_list)
        current_S_arr = S.copy()
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Check if x can be added
            # 1. 3x
            if sums_mask[3*x]:
                continue
            
            # 2. 2x + s for s in current_S
            # Vectorized: new_sums_2 = 2*x + current_S_arr
            if len(current_S_arr) > 0:
                new_sums_2 = 2*x + current_S_arr
                if np.any(sums_mask[new_sums_2]):
                    continue
            else:
                new_sums_2 = np.array([], dtype=np.int64)
            
            # 3. x + s_i + s_j for s_i, s_j in current_S
            # Vectorized: outer sum
            if len(current_S_arr) > 1:
                pair_sums = current_S_arr[:, None] + current_S_arr[None, :]
                new_sums_3 = x + pair_sums
                # Flatten and check
                new_sums_3_flat = new_sums_3.flatten()
                if np.any(sums_mask[new_sums_3_flat]):
                    continue
            elif len(current_S_arr) == 1:
                # Only one element s, pair is (s,s)
                new_sums_3_flat = np.array([x + 2*current_S_arr[0]], dtype=np.int64)
                if np.any(sums_mask[new_sums_3_flat]):
                    continue
            else:
                new_sums_3_flat = np.array([], dtype=np.int64)
            
            # All checks passed. Add x.
            current_S.append(x)
            current_S_arr = np.append(current_S_arr, x)
            
            # Update sums_mask
            sums_mask[3*x] = True
            if len(new_sums_2) > 0:
                sums_mask[new_sums_2] = True
            if len(new_sums_3_flat) > 0:
                sums_mask[new_sums_3_flat] = True
            
            # Update best
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_set = list(current_S)
                
            # Optional: Early stop if we hit a high target? 
            # The max possible is around 10-20 for this constraint? 
            # Actually, B_3[1] sets are dense. 
            # Let's just run until deadline or candidates exhausted.

    # Define order functions
    def order_asc(cands):
        return sorted(cands)
    
    def order_desc(cands):
        return sorted(cands, reverse=True)
    
    def order_random(cands):
        c = cands[:]
        random.shuffle(c)
        return c
    
    # Skeleton Generators
    def skeleton_powers_of_3():
        # 1, 3, 9, 27, 81, 243, 729, 2187
        return [1, 3, 9, 27, 81, 243, 729, 2187]
        
    def skeleton_powers_of_4():
        # 1, 4, 16, 64, 256, 1024, 4096
        return [1, 4, 16, 64, 256, 1024, 4096]
        
    def skeleton_random_geometric(base=4, jitter=2):
        skel = []
        curr = 1
        while curr <= n:
            skel.append(curr)
            # Next is curr * base + random jitter
            curr = curr * base + random.randint(0, jitter)
        return skel

    # Run multiple strategies
    # 1. Pure Ascending Greedy (Standard)
    # This is equivalent to skeleton=[] and order_asc
    # But our try_fill requires a valid skeleton. An empty set is valid.
    try_fill([], order_asc, "asc")
    
    if time.time() > deadline:
        return best_set
        
    # 2. Pure Descending Greedy
    try_fill([], order_desc, "desc")
    
    if time.time() > deadline:
        return best_set
        
    # 3. Random Greedy from Empty Set
    # Run a few times
    for i in range(10):
        if time.time() > deadline:
            break
        try_fill([], order_random, f"rand_empty_{i}")
        
    if time.time() > deadline:
        return best_set
        
    # 4. Skeleton-based Fills
    # Start with a geometric skeleton, then fill randomly/asc/desc
    skeletons = [
        skeleton_powers_of_3(),
        skeleton_powers_of_4(),
        skeleton_random_geometric(4, 1),
        skeleton_random_geometric(3, 1),
        skeleton_random_geometric(5, 2),
        skeleton_random_geometric(4, 3),
    ]
    
    for skel in skeletons:
        # Ensure skeleton is valid and within bounds
        skel = [x for x in skel if 1 <= x <= n]
        if not skel:
            continue
            
        # Try different fill orders for this skeleton
        # Ascending fill
        try_fill(skel, order_asc, "skel_asc")
        if time.time() > deadline:
            break
            
        # Descending fill
        try_fill(skel, order_desc, "skel_desc")
        if time.time() > deadline:
            break
            
        # Random fill
        try_fill(skel, order_random, "skel_rand")
        if time.time() > deadline:
            break

    return best_set
