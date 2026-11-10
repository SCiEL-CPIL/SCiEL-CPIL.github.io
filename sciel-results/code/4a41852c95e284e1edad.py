def construct():
    import time
    from math import gcd

    deadline = time.time() + 85  # 85s out of 90s budget
    
    N = 256
    nums = list(range(2, 257))
    n = len(nums)  # 255
    
    # Precompute conflict masks
    # conflict_mask[i] is an integer with bit j set if nums[i] and nums[j] conflict
    conflict_mask = [0] * n
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    weights = [1.0 / nums[i] for i in range(n)]
    
    def get_score(mask):
        # Sum weights of bits set in mask
        # Since n is small, we can iterate
        s = 0.0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    def can_add(idx, mask):
        return (conflict_mask[idx] & mask) == 0

    # Greedy Initialization
    # Try a few orderings
    best_mask = 0
    best_score = 0.0
    
    # Ordering 1: Weight desc
    order1 = sorted(range(n), key=lambda i: -weights[i])
    # Ordering 2: Conflict count asc
    conflict_counts = [bin(conflict_mask[i]).count('1') for i in range(n)]
    order2 = sorted(range(n), key=lambda i: (conflict_counts[i], -weights[i]))
    
    for order in [order1, order2]:
        mask = 0
        for i in order:
            if can_add(i, mask):
                mask |= (1 << i)
        score = get_score(mask)
        if score > best_score:
            best_score = score
            best_mask = mask

    # Local Search with 1-swap and 2-swap
    current_mask = best_mask
    current_score = best_score
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try 1-swaps
        # For each i in S, find best j in U such that j compatible with S\{i} and w_j > w_i
        # Actually, just find any j that improves.
        # To be efficient, we can iterate i, then iterate j in U sorted by weight desc?
        # Or just iterate all pairs.
        
        # Let's do a more systematic local search.
        # We will maintain a list of "moves" or just do steepest ascent.
        
        # 1-Swap
        for i in range(n):
            if not (current_mask & (1 << i)):
                continue
            # Remove i
            mask_without_i = current_mask ^ (1 << i)
            # Find best j in U that can be added to mask_without_i
            best_j = -1
            best_w_j = -1.0
            # Iterate over all j not in current_mask
            # Optimization: Only consider j with weight > weights[i]
            # But we need to check compatibility.
            for j in range(n):
                if current_mask & (1 << j):
                    continue
                if weights[j] <= weights[i]:
                    continue
                if can_add(j, mask_without_i):
                    # Valid 1-swap
                    new_score = current_score - weights[i] + weights[j]
                    if new_score > current_score + 1e-15:
                        current_mask = mask_without_i | (1 << j)
                        current_score = new_score
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
            
        # 2. Try 2-swaps
        # For each pair i, k in S, find best pair j, l in U
        # such that j, l compatible with S\{i,k} and with each other, and w_j+w_l > w_i+w_k
        
        # To speed up, we can precompute the list of indices in S and U
        # But n is small, so we can just loop.
        
        found_2swap = False
        # Iterate i in S
        for i in range(n):
            if not (current_mask & (1 << i)):
                continue
            for k in range(i + 1, n):
                if not (current_mask & (1 << k)):
                    continue
                mask_rem = current_mask ^ (1 << i) ^ (1 << k)
                target_w = weights[i] + weights[k]
                
                # Find candidates in U compatible with mask_rem
                # We want the two highest weight candidates that are compatible with each other.
                # Let's collect valid candidates
                valid_candidates = []
                for j in range(n):
                    if current_mask & (1 << j):
                        continue
                    if can_add(j, mask_rem):
                        valid_candidates.append(j)
                        
                if len(valid_candidates) < 2:
                    continue
                    
                # Sort by weight descending
                valid_candidates.sort(key=lambda x: -weights[x])
                
                # Find first compatible pair
                # Since we want max sum, check top elements
                # Worst case O(M^2) where M is len(valid_candidates)
                # M is likely small (most nodes conflict with many)
                found_pair = False
                for a_idx in range(len(valid_candidates)):
                    j = valid_candidates[a_idx]
                    if weights[j] + weights[valid_candidates[-1]] <= target_w:
                        # Even with the smallest possible partner, we can't beat target?
                        # Actually, we just need ANY pair > target.
                        # If top 2 sum <= target, no pair will work?
                        # Yes, because weights are positive and sorted.
                        break
                    for b_idx in range(a_idx + 1, len(valid_candidates)):
                        l = valid_candidates[b_idx]
                        # Check if j and l are compatible
                        if (conflict_mask[j] & (1 << l)) == 0:
                            new_score = current_score - target_w + weights[j] + weights[l]
                            if new_score > current_score + 1e-15:
                                current_mask = mask_rem | (1 << j) | (1 << l)
                                current_score = new_score
                                improved = True
                                found_pair = True
                                break
                    if found_pair:
                        break
                if found_pair:
                    break
            if found_pair:
                break
        
        if found_pair:
            improved = True

    # Extract result
    result = []
    for i in range(n):
        if current_mask & (1 << i):
            result.append(nums[i])
            
    return result
