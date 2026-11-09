import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 256
    small_nums = list(range(2, 129))  # 2 to 128
    large_nums = list(range(129, 257)) # 129 to 256
    
    n_small = len(small_nums)
    n_large = len(large_nums)
    
    # Weights
    small_weights = [1.0 / x for x in small_nums]
    large_weights = [1.0 / x for x in large_nums]
    
    # Precompute LCM conflicts within Small
    # small_conflict[i][j] is True if lcm(small[i], small[j]) <= 256
    small_conflict = [[False] * n_small for _ in range(n_small)]
    for i in range(n_small):
        a = small_nums[i]
        for j in range(i + 1, n_small):
            b = small_nums[j]
            if math.lcm(a, b) <= N:
                small_conflict[i][j] = True
                small_conflict[j][i] = True
                
    # Precompute which Large elements are blocked by each Small element
    # blocked_large_mask[i] is a bitmask of indices in large_nums that conflict with small_nums[i]
    blocked_large_mask = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        mask = 0
        for j in range(n_large):
            b = large_nums[j]
            if math.lcm(a, b) <= N:
                mask |= (1 << j)
        blocked_large_mask[i] = mask
        
    # Precompute weight of large subset given a mask
    # Since n_large is 128, we can't precompute all 2^128.
    # But we can compute weight of a mask quickly if we have prefix sums? No, arbitrary mask.
    # We can precompute weight of each bit and sum them up.
    # Or, since we only add/remove one element at a time in local search, 
    # we can maintain the current blocked mask and its weight.
    
    total_large_weight = sum(large_weights)
    
    def get_large_weight_from_mask(mask):
        # Sum weights of set bits in mask
        w = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            w += large_weights[idx]
            m ^= lsb
        return w

    best_score = total_large_weight # Start with X = empty, Y = all L
    best_x_mask = 0 # Bitmask of selected small elements
    
    def evaluate_x_mask(x_mask):
        # Calculate score for a given set of small elements
        # Score = W(X) - W(B(X)) + Total_Large_Weight
        w_x = 0.0
        blocked_mask = 0
        m = x_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            w_x += small_weights[idx]
            blocked_mask |= blocked_large_mask[idx]
            m ^= lsb
            
        w_blocked = get_large_weight_from_mask(blocked_mask)
        return w_x - w_blocked + total_large_weight

    # Local Search
    # Start with empty X
    current_x_mask = 0
    current_score = evaluate_x_mask(current_x_mask)
    
    improved = True
    iterations = 0
    max_iterations = 10000
    
    while improved and iterations < max_iterations and time.time() < deadline:
        improved = False
        iterations += 1
        
        # Try adding a small element
        best_add_gain = 0.0
        best_add_idx = -1
        for i in range(n_small):
            if current_x_mask & (1 << i):
                continue
            # Check if i is compatible with all current small elements
            compatible = True
            m = current_x_mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if small_conflict[i][j]:
                    compatible = False
                    break
                m ^= lsb
            if not compatible:
                continue
            
            # Calculate new score
            # Delta = small_weights[i] - (New_Blocked_Weight - Old_Blocked_Weight)
            # We need to know which new large elements are blocked.
            # new_blocked = old_blocked | blocked_large_mask[i]
            # We don't have old_blocked stored, but we can recompute or store it.
            # For efficiency in this loop, let's just recompute score if candidate looks good?
            # Or store current_blocked_mask.
            
            # Let's store current_blocked_mask outside the loop for efficiency
            pass # Placeholder for optimization
            
        # Optimization: Store current state variables
        # Re-implementing loop with stored state
        
    # Re-do local search with proper state tracking
    
    current_x_mask = 0
    current_blocked_large = 0
    current_w_x = 0.0
    current_w_blocked = 0.0
    current_score = total_large_weight
    
    best_x_mask = 0
    best_score = current_score
    
    def update_best():
        nonlocal best_x_mask, best_score
        if current_score > best_score:
            best_score = current_score
            best_x_mask = current_x_mask

    update_best()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try Add
        best_add_gain = 0.0
        best_add_idx = -1
        
        for i in range(n_small):
            if current_x_mask & (1 << i):
                continue
            
            # Check compatibility with current X
            compatible = True
            m = current_x_mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if small_conflict[i][j]:
                    compatible = False
                    break
                m ^= lsb
            if not compatible:
                continue
            
            # Calculate gain
            # New blocked bits = current_blocked_large | blocked_large_mask[i]
            # New blocked weight = current_w_blocked + weight(new bits)
            new_blocked_bits = current_blocked_large | blocked_large_mask[i]
            added_blocked_bits = new_blocked_bits & ~current_blocked_large
            
            # Calculate weight of added_blocked_bits
            w_added_blocked = 0.0
            m = added_blocked_bits
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                w_added_blocked += large_weights[idx]
                m ^= lsb
            
            gain = small_weights[i] - w_added_blocked
            
            if gain > best_add_gain:
                best_add_gain = gain
                best_add_idx = i
                
        if best_add_idx != -1:
            i = best_add_idx
            current_x_mask |= (1 << i)
            new_blocked_bits = current_blocked_large | blocked_large_mask[i]
            added_blocked_bits = new_blocked_bits & ~current_blocked_large
            w_added_blocked = 0.0
            m = added_blocked_bits
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                w_added_blocked += large_weights[idx]
                m ^= lsb
            
            current_blocked_large = new_blocked_bits
            current_w_x += small_weights[i]
            current_w_blocked += w_added_blocked
            current_score += best_add_gain
            improved = True
            update_best()
            continue # Restart search with new state
            
        # 2. Try Remove
        best_remove_gain = 0.0
        best_remove_idx = -1
        
        m = current_x_mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            m ^= lsb
            
            # Gain from removing i
            # We lose small_weights[i]
            # We unblock some large elements
            # Unblocked bits = blocked_large_mask[i] & ~ (blocked by other x in X)
            # This is hard to compute directly without recomputing union.
            # Approximation: Assume unblocking all bits in blocked_large_mask[i] that are currently blocked.
            # But some might be blocked by others.
            # Let's just recompute score for candidate removal? 
            # Since n_small is small, we can afford to be a bit less efficient here.
            
            # Calculate new state if we remove i
            new_x_mask = current_x_mask & ~(1 << i)
            
            # Recompute blocked mask for new_x_mask
            new_blocked = 0
            mm = new_x_mask
            while mm:
                lsb2 = mm & -mm
                idx2 = lsb2.bit_length() - 1
                new_blocked |= blocked_large_mask[idx2]
                mm ^= lsb2
                
            # Weight of new blocked
            w_new_blocked = get_large_weight_from_mask(new_blocked)
            w_new_x = current_w_x - small_weights[i]
            
            new_score = w_new_x - w_new_blocked + total_large_weight
            gain = new_score - current_score
            
            if gain > best_remove_gain:
                best_remove_gain = gain
                best_remove_idx = i
                
        if best_remove_idx != -1:
            i = best_remove_idx
            current_x_mask &= ~(1 << i)
            
            # Recompute state
            current_w_x -= small_weights[i]
            new_blocked = 0
            m = current_x_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                new_blocked |= blocked_large_mask[idx]
                m ^= lsb
            current_blocked_large = new_blocked
            current_w_blocked = get_large_weight_from_mask(current_blocked_large)
            current_score = current_w_x - current_w_blocked + total_large_weight
            improved = True
            update_best()
            continue
            
        # 3. Try Swap (Remove i, Add j)
        # This is more complex. We can skip for now if Add/Remove is sufficient.
        # Given the structure, Add/Remove might get stuck in local minima.
        # But let's try a simple random restart if stuck.
        
        if not improved:
            # Random restart with a random initial X?
            # Or just break.
            break
            
    # Construct final set
    final_set = []
    
    # Add all Large elements not blocked by best_x_mask
    blocked_mask_final = 0
    m = best_x_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        blocked_mask_final |= blocked_large_mask[idx]
        m ^= lsb
        
    for j in range(n_large):
        if not (blocked_mask_final & (1 << j)):
            final_set.append(large_nums[j])
            
    # Add selected Small elements
    m = best_x_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        final_set.append(small_nums[idx])
        m ^= lsb
        
    if not final_set:
        return [2]
        
    return final_set
