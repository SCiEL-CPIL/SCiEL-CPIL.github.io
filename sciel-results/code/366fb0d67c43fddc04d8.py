def construct():
    import math
    
    L_vals = list(range(129, 257))
    V_vals = list(range(2, 129))
    
    n_L = len(L_vals)
    n_V = len(V_vals)
    
    # Precompute weights for L
    L_weights = [1.0 / x for x in L_vals]
    
    # Precompute M_u: bitmask of C_u in L
    # M_u[i] is 1 if L_vals[i] is in C_u
    M_u = [0] * n_V
    for i, u in enumerate(V_vals):
        mask = 0
        for j, y in enumerate(L_vals):
            if math.lcm(u, y) <= 256:
                mask |= (1 << j)
        M_u[i] = mask
        
    # Precompute neighbor masks for G_V
    # N_u[i] is bitmask of neighbors of V_vals[i] in V
    N_u = [0] * n_V
    for i, u in enumerate(V_vals):
        mask = 0
        for j, v in enumerate(V_vals):
            if i != j and math.lcm(u, v) <= 256:
                mask |= (1 << j)
        N_u[i] = mask
        
    # Helper to calculate weight of a mask in L
    def get_weight(mask):
        w = 0.0
        m = mask
        j = 0
        while m:
            if m & 1:
                w += L_weights[j]
            m >>= 1
            j += 1
        return w

    # Initial best
    # Start with empty A, so L is fully present
    # Score = sum(1/y for y in L)
    base_score = sum(L_weights)
    best_score = base_score
    
    # B&B
    # State: v_mask, l_mask, sum_w, l_weight
    # v_mask: available nodes in V
    # l_mask: covered nodes in L
    # sum_w: sum of weights of selected A
    # l_weight: weight of covered L nodes
    
    # Initial state
    init_v_mask = (1 << n_V) - 1
    init_l_mask = 0
    init_sum_w = 0.0
    init_l_weight = 0.0
    
    # We want to maximize: sum_w - l_weight
    # But we add to base_score?
    # Total Score = base_score - l_weight + sum_w
    # So we maximize (sum_w - l_weight)
    
    # Optimization: Precompute weight of single bits for L?
    # L_weights is already there.
    
    # To speed up get_weight for small added masks, we can just sum bits.
    
    def dfs(v_mask, l_mask, sum_w, l_weight):
        nonlocal best_score
        
        if v_mask == 0:
            score = sum_w - l_weight
            if score > best_score:
                best_score = score
            return
            
        # Upper Bound
        # Calculate optimistic gain
        # For each u in v_mask, max gain is w_u - weight(M_u & ~l_mask)
        # If gain <= 0, it doesn't help to include it (assuming no synergy, which there isn't for weights, but there is for cost sharing? 
        # Cost is additive over union. If I include u and v, cost is weight(C_u U C_v).
        # Gain(u) + Gain(v) = w_u + w_v - weight(C_u) - weight(C_v).
        # Actual Gain(u, v) = w_u + w_v - weight(C_u U C_v).
        # Since weight(C_u U C_v) <= weight(C_u) + weight(C_v), Actual Gain >= Sum of Individual Gains?
        # No. weight(C_u U C_v) = weight(C_u) + weight(C_v) - weight(C_u & C_v).
        # So Actual Gain = w_u + w_v - weight(C_u) - weight(C_v) + weight(C_u & C_v).
        # Individual Gains sum = w_u - weight(C_u & ~l_mask) + w_v - weight(C_v & ~l_mask).
        # This is getting complicated. 
        # Simple UB: sum of max(0, w_u - weight(M_u & ~l_mask)) for u in v_mask.
        # This assumes we can pick all of them and costs are disjoint. 
        # Since costs are submodular (union weight is submodular?), the sum of individual marginal gains is an upper bound on the total marginal gain?
        # Yes, for maximizing a function that is the sum of weights minus a submodular cost, the greedy choice gives a lower bound, but the sum of individual best gains is a valid upper bound for the remaining potential.
        
        opt_gain = 0.0
        m = v_mask
        i = 0
        while m:
            if m & 1:
                u_idx = i
                w_u = 1.0 / V_vals[u_idx]
                # Cost of adding u given current l_mask
                added_mask = M_u[u_idx] & ~l_mask
                cost = get_weight(added_mask)
                gain = w_u - cost
                if gain > 0:
                    opt_gain += gain
            m >>= 1
            i += 1
            
        current_val = sum_w - l_weight
        ub = current_val + opt_gain
        if ub <= best_score:
            return
            
        # Branch
        # Pick first set bit (smallest index, highest weight)
        i = 0
        m = v_mask
        while m and not (m & 1):
            m >>= 1
            i += 1
        # Now m & 1 is True, i is the index
        u_idx = i
        u_val = V_vals[u_idx]
        w_u = 1.0 / u_val
        
        # Option 1: Exclude u
        dfs(v_mask ^ (1 << u_idx), l_mask, sum_w, l_weight)
        
        # Option 2: Include u
        # Update l_mask
        new_l_mask = l_mask | M_u[u_idx]
        added_l_mask = new_l_mask ^ l_mask
        new_l_weight = l_weight + get_weight(added_l_mask)
        
        # Update v_mask
        # Remove u and its neighbors
        new_v_mask = v_mask & ~N_u[u_idx] & ~(1 << u_idx)
        
        dfs(new_v_mask, new_l_mask, sum_w + w_u, new_l_weight)

    dfs(init_v_mask, init_l_mask, init_sum_w, init_l_weight)
    
    # best_score is the gain over base_score
    # We need to return the subset.
    # Wait, the task asks to return the subset.
    # My B&B only finds the max score. I need to reconstruct the set.
    
    # I should modify B&B to store the best set.
    # Or, since I only need the value for the score in the prompt?
    # "Return nonempty subset... Maximize exact reciprocal sum"
    # The evaluator measures the raw value.
    # I need to return the list.
    
    # Let's add `best_A` to nonlocal.
    
    # Re-implement with tracking.
    
    best_A = []
    
    def dfs_track(v_mask, l_mask, sum_w, l_weight, current_A):
        nonlocal best_score, best_A
        
        if v_mask == 0:
            score = sum_w - l_weight
            if score > best_score:
                best_score = score
                best_A = list(current_A)
            return
            
        # Upper Bound
        opt_gain = 0.0
        m = v_mask
        i = 0
        while m:
            if m & 1:
                u_idx = i
                w_u = 1.0 / V_vals[u_idx]
                added_mask = M_u[u_idx] & ~l_mask
                cost = get_weight(added_mask)
                gain = w_u - cost
                if gain > 0:
                    opt_gain += gain
            m >>= 1
            i += 1
            
        current_val = sum_w - l_weight
        ub = current_val + opt_gain
        if ub <= best_score:
            return
            
        # Branch
        i = 0
        m = v_mask
        while m and not (m & 1):
            m >>= 1
            i += 1
        u_idx = i
        u_val = V_vals[u_idx]
        w_u = 1.0 / u_val
        
        # Option 1: Exclude u
        dfs_track(v_mask ^ (1 << u_idx), l_mask, sum_w, l_weight, current_A)
        
        # Option 2: Include u
        new_l_mask = l_mask | M_u[u_idx]
        added_l_mask = new_l_mask ^ l_mask
        new_l_weight = l_weight + get_weight(added_l_mask)
        new_v_mask = v_mask & ~N_u[u_idx] & ~(1 << u_idx)
        
        current_A.append(u_val)
        dfs_track(new_v_mask, new_l_mask, sum_w + w_u, new_l_weight, current_A)
        current_A.pop()

    dfs_track(init_v_mask, init_l_mask, init_sum_w, init_l_weight, [])
    
    # Construct final set
    # A is best_A
    # L_remaining is L_vals where bit is NOT in l_mask corresponding to best_A
    # Wait, I didn't store the final l_mask.
    # I can recompute it or store it.
    # Let's store best_l_mask.
    
    best_l_mask = 0
    
    def dfs_track2(v_mask, l_mask, sum_w, l_weight, current_A):
        nonlocal best_score, best_A, best_l_mask
        
        if v_mask == 0:
            score = sum_w - l_weight
            if score > best_score:
                best_score = score
                best_A = list(current_A)
                best_l_mask = l_mask
            return
            
        opt_gain = 0.0
        m = v_mask
        i = 0
        while m:
            if m & 1:
                u_idx = i
                w_u = 1.0 / V_vals[u_idx]
                added_mask = M_u[u_idx] & ~l_mask
                cost = get_weight(added_mask)
                gain = w_u - cost
                if gain > 0:
                    opt_gain += gain
            m >>= 1
            i += 1
            
        current_val = sum_w - l_weight
        ub = current_val + opt_gain
        if ub <= best_score:
            return
            
        i = 0
        m = v_mask
        while m and not (m & 1):
            m >>= 1
            i += 1
        u_idx = i
        u_val = V_vals[u_idx]
        w_u = 1.0 / u_val
        
        dfs_track2(v_mask ^ (1 << u_idx), l_mask, sum_w, l_weight, current_A)
        
        new_l_mask = l_mask | M_u[u_idx]
        added_l_mask = new_l_mask ^ l_mask
        new_l_weight = l_weight + get_weight(added_l_mask)
        new_v_mask = v_mask & ~N_u[u_idx] & ~(1 << u_idx)
        
        current_A.append(u_val)
        dfs_track2(new_v_mask, new_l_mask, sum_w + w_u, new_l_weight, current_A)
        current_A.pop()

    dfs_track2(init_v_mask, init_l_mask, init_sum_w, init_l_weight, [])
    
    # Build result
    result = list(best_A)
    for i, y in enumerate(L_vals):
        if not (best_l_mask & (1 << i)):
            result.append(y)
            
    return result
