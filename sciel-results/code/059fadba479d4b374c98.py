"""Rewritten program: Hybrid Exact/Heuristic Solver for Max Weight Independent Set."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n_max = 256
    # Partition: Small numbers (2-16) have high weight and dense conflicts.
    # Large numbers (17-256) have lower weight and sparser conflicts.
    small_nums = list(range(2, 17))
    large_nums = list(range(17, 257))
    
    n_small = len(small_nums)
    n_large = len(large_nums)
    
    # Map values to indices
    small_idx = {v: i for i, v in enumerate(small_nums)}
    large_idx = {v: i for i, v in enumerate(large_nums)}
    
    # Weights
    w_small = [1.0 / v for v in small_nums]
    w_large = [1.0 / v for v in large_nums]
    
    # Precompute conflicts
    # conf_small[i] is bitmask of small nodes conflicting with small_nums[i]
    conf_small = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        for j in range(i + 1, n_small):
            b = small_nums[j]
            if a * b // gcd(a, b) <= n_max:
                conf_small[i] |= (1 << j)
                conf_small[j] |= (1 << i)
                
    # conf_large[j] is bitmask of large nodes conflicting with large_nums[j]
    conf_large = [0] * n_large
    for i in range(n_large):
        a = large_nums[i]
        for j in range(i + 1, n_large):
            b = large_nums[j]
            if a * b // gcd(a, b) <= n_max:
                conf_large[i] |= (1 << j)
                conf_large[j] |= (1 << i)
                
    # block_large_by_small[i] is bitmask of large nodes blocked by small_nums[i]
    # i.e., large node j is blocked if lcm(small_nums[i], large_nums[j]) <= 256
    block_large_by_small = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        mask = 0
        for j in range(n_large):
            b = large_nums[j]
            if a * b // gcd(a, b) <= n_max:
                mask |= (1 << j)
        block_large_by_small[i] = mask

    # Fast Solver for Large Numbers
    # Input: free_mask (bitmask of large nodes that are NOT blocked by current small selection)
    # Returns: (score, selected_mask)
    def solve_large(free_mask):
        if free_mask == 0:
            return 0.0, 0
        
        # Greedy Initialization
        # Sort large indices by weight descending
        # We only consider nodes in free_mask
        # To make this fast, we can just iterate through pre-sorted indices
        # and check if they are in free_mask and not conflicting with selected.
        
        # Pre-sorted indices of large nodes by weight (value ascending)
        # This is static
        large_order = sorted(range(n_large), key=lambda x: -w_large[x])
        
        selected_mask = 0
        score = 0.0
        
        for idx in large_order:
            if not (free_mask & (1 << idx)):
                continue
            if (conf_large[idx] & selected_mask) == 0:
                selected_mask |= (1 << idx)
                score += w_large[idx]
        
        # Local Search: 1-swap
        # Try to swap out a selected node i and in an unselected node j
        # Gain = w_large[j] - w_large[i] > 0
        # Condition: j compatible with (selected \ {i})
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Iterate selected nodes
            m = selected_mask
            while m:
                lsb_i = m & -m
                i = lsb_i.bit_length() - 1
                m ^= lsb_i
                
                # Potential removal of i
                temp_mask = selected_mask & ~lsb_i
                # Find best j in free_mask \ selected_mask that is compatible with temp_mask
                # and has weight > w_large[i]
                # To speed up, iterate unselected candidates in large_order
                best_j = -1
                best_gain = 0
                best_w = 0
                
                # Optimization: Only consider j if w_large[j] > w_large[i]
                for j in large_order:
                    if w_large[j] <= w_large[i]:
                        break # Since sorted descending, no further j will have higher weight
                    if selected_mask & (1 << j):
                        continue
                    if not (free_mask & (1 << j)):
                        continue
                    if (conf_large[j] & temp_mask) == 0:
                        gain = w_large[j] - w_large[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_j = j
                            best_w = w_large[j]
                
                if best_j != -1:
                    selected_mask = temp_mask | (1 << best_j)
                    score += best_gain
                    improved = True
                    break # Restart while loop for m iteration to reflect new state
            
            if time.time() > deadline:
                break
                
        return score, selected_mask

    # B&B for Small Numbers
    best_total_score = 0.0
    best_small_mask = 0
    best_large_mask = 0
    
    # Sort small nodes by weight descending for B&B
    small_order = sorted(range(n_small), key=lambda x: -w_small[x])
    
    # Precompute suffix sums of small weights for upper bound
    # suffix_sum[i] = sum of weights of small nodes from index i to end in small_order
    suffix_sum = [0.0] * (n_small + 1)
    for i in range(n_small - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + w_small[small_order[i]]
    
    # Precompute greedy upper bound for large part?
    # For a given forbidden mask, the max score is bounded by sum of all free weights.
    # A tighter bound is the greedy score, but calculating it at every node is expensive.
    # We will use sum of weights of free nodes as a simple upper bound.
    
    def search_small(idx, current_mask, current_score, forbidden_large_mask):
        nonlocal best_total_score, best_small_mask, best_large_mask
        
        if time.time() > deadline:
            return
            
        # Calculate upper bound
        # 1. Sum of remaining small weights
        ub_small = suffix_sum[idx]
        
        # 2. Upper bound for large part
        # Free large nodes = all large nodes not in forbidden_large_mask
        # Note: forbidden_large_mask accumulates blocks from selected smalls.
        # We need to know which large nodes are still available.
        # all_large_mask is (1 << n_large) - 1
        all_large_mask = (1 << n_large) - 1
        free_large_mask = all_large_mask & ~forbidden_large_mask
        
        # Simple upper bound for large: sum of weights of free_large_mask
        # This is loose but fast.
        # Better: Greedy score of free_large_mask. But that's O(N^2).
        # Let's try to be a bit smarter.
        # If idx == n_small, we must solve large exactly (heuristic).
        if idx == n_small:
            l_score, l_mask = solve_large(free_large_mask)
            total = current_score + l_score
            if total > best_total_score:
                best_total_score = total
                best_small_mask = current_mask
                best_large_mask = l_mask
            return
        
        # Pruning
        # Upper bound = current_score + ub_small + sum(w_large[j] for j in free_large_mask)
        # Calculating sum of free large weights quickly:
        # We can precompute weight sums for bitmasks? No, too large.
        # We can just iterate? N_large is 240. Iterating 240 bits is fast.
        # But doing it at every node might be slow.
        # Let's approximate: max possible large score is sum of all large weights.
        # Let's use a slightly tighter bound: Greedy on free_large_mask?
        # If we call solve_large at every node, it's too slow.
        # We will use the sum of weights of free nodes as the bound.
        
        sum_free_large = 0.0
        m = free_large_mask
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            sum_free_large += w_large[j]
            m ^= lsb
            
        ub_total = current_score + ub_small + sum_free_large
        
        if ub_total <= best_total_score + 1e-9:
            return
            
        # Branch on small_order[idx]
        i = small_order[idx]
        
        # Option 1: Include i
        # Check if i conflicts with current_mask
        if (conf_small[i] & current_mask) == 0:
            new_mask = current_mask | (1 << i)
            new_forbidden = forbidden_large_mask | block_large_by_small[i]
            search_small(idx + 1, new_mask, current_score + w_small[i], new_forbidden)
            
        # Option 2: Exclude i
        search_small(idx + 1, current_mask, current_score, forbidden_large_mask)

    # Start search
    search_small(0, 0, 0.0, 0)
    
    # Extract result
    result = []
    
    m = best_small_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        result.append(small_nums[i])
        m ^= lsb
        
    m = best_large_mask
    while m:
        lsb = m & -m
        j = lsb.bit_length() - 1
        result.append(large_nums[j])
        m ^= lsb
        
    return result

# EVOLVE-BLOCK-END
