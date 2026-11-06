import time
import random

def _can_add_with_dp(dp, current_set, cand, n):
    """
    Check if cand can be added to current_set given the DP table for current_set.
    Returns True if safe. n = len(current_set)
    """
    if n == 0:
        return True
    
    min_S = min(current_set)
    max_S = max(current_set)
    
    # 1. cand is not mean of any subset of current_set (size >= 2)
    for k in range(2, n + 1):
        target = k * cand
        min_sum_k = k * min_S
        max_sum_k = k * max_S
        if target < min_sum_k or target > max_sum_k:
            continue
        if (dp[k] >> target) & 1:
            return False
            
    # 2. No a in current_set is mean of subset of (current_set U {cand}) including cand.
    # For each a in current_set, check if there exists T subset of current_set, |T|>=1
    # such that sum(T) + cand = (|T|+1) * a
    # i.e., sum(T) = a*(|T|+1) - cand
    # Optimize: skip if target out of range
    for a in current_set:
        for k in range(1, n + 1):
            target = a * (k + 1) - cand
            min_sum_k = k * min_S
            max_sum_k = k * max_S
            if target < min_sum_k or target > max_sum_k:
                continue
            if (dp[k] >> target) & 1:
                return False
    return True

def _update_dp(dp, cand, k_prev):
    """
    Update DP table in-place to include cand.
    k_prev = size of set before adding cand.
    """
    for k in range(k_prev + 1, 0, -1):
        dp[k] |= (dp[k - 1] << cand)

def _remove_from_dp(dp, cand, k_prev):
    """
    Remove cand from DP table in-place.
    k_prev = size of set before removal (i.e., current size).
    """
    for k in range(k_prev, 0, -1):
        # dp[k] had contributions from dp[k-1] with cand added
        # We need to subtract those contributions
        # But we can't easily subtract bitsets... 
        # Instead, we'll rebuild. This is a limitation.
        pass

def _count_conflicts_fast(dp, current_set, cand, n):
    """
    Count how many subset-mean violations would be created if cand were added.
    Returns the count (0 means safe to add).
    """
    if n == 0:
        return 0
    
    min_S = min(current_set)
    max_S = max(current_set)
    count = 0
    
    # 1. cand is mean of any subset of current_set (size >= 2)
    for k in range(2, n + 1):
        target = k * cand
        min_sum_k = k * min_S
        max_sum_k = k * max_S
        if target < min_sum_k or target > max_sum_k:
            continue
        if (dp[k] >> target) & 1:
            count += 1
            if count > 0:
                # Early exit for speed - we just need to know it's > 0
                pass
    
    # 2. No a in current_set is mean of subset including cand
    for a in current_set:
        for k in range(1, n + 1):
            target = a * (k + 1) - cand
            min_sum_k = k * min_S
            max_sum_k = k * max_S
            if target < min_sum_k or target > max_sum_k:
                continue
            if (dp[k] >> target) & 1:
                count += 1
    
    return count

def _build_dp_from_set(s):
    """Build DP table from a set of numbers."""
    max_dp_size = len(s) + 1
    dp = [0] * (max_dp_size + 1)
    dp[0] = 1
    for i, x in enumerate(s):
        _update_dp(dp, x, i)
    return dp

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_len = 0
    
    nums = list(range(1, 513))
    
    # Strategy 1: Greedy with multiple orderings
    orders = []
    
    # Edge first
    edge_order = []
    l, r = 1, 512
    while l <= r:
        edge_order.append(l)
        if l != r:
            edge_order.append(r)
        l += 1
        r -= 1
    orders.append(edge_order)
    
    # Ascending, descending
    orders.append(list(range(1, 513)))
    orders.append(list(range(512, 0, -1)))
    
    # Random orders with different seeds
    for seed in range(150):
        rnd = random.Random(seed)
        rnd_order = list(range(1, 513))
        rnd.shuffle(rnd_order)
        orders.append(rnd_order)
        
    # Weighted random: bias towards edges
    for seed in range(150):
        rnd = random.Random(seed + 1000)
        nums = list(range(1, 513))
        keys = []
        for x in nums:
            dist = abs(x - 256.5)
            key = dist * 0.7 + rnd.random() * 256
            keys.append(key)
        paired = sorted(zip(keys, nums), key=lambda t: -t[0])
        edge_biased = [p[1] for p in paired]
        orders.append(edge_biased)

    for order in orders:
        if time.time() > deadline - 3.0:
            break
            
        current_set_list = []
        max_dp_size = 50
        dp = [0] * (max_dp_size + 1)
        dp[0] = 1
        
        for cand in order:
            if time.time() > deadline - 2.0:
                break
            
            n = len(current_set_list)
            if _can_add_with_dp(dp, current_set_list, cand, n):
                current_set_list.append(cand)
                _update_dp(dp, cand, n)
                
        if len(current_set_list) > best_len:
            best_set = current_set_list[:]
            best_len = len(best_set)
            
            if best_len >= 48:
                return best_set

    # Strategy 2: Conflict-driven local search with swaps
    if time.time() < deadline - 2.0 and best_len < 48:
        best_set = sorted(best_set)
        best_set_set = set(best_set)
        dp_best = _build_dp_from_set(best_set)
        n = len(best_set)
        
        # Precompute conflict counts for all unused candidates
        unused = [x for x in range(1, 513) if x not in best_set_set]
        
        improved = True
        max_swaps = 5000
        swap_count = 0
        
        while improved and time.time() < deadline - 2.0 and swap_count < max_swaps:
            improved = False
            swap_count += 1
            n = len(best_set)
            dp_best = _build_dp_from_set(best_set)
            
            # Try to find a beneficial swap: remove one element, add another
            # First, try simple additions
            for cand in unused:
                if cand in best_set_set:
                    continue
                if time.time() > deadline - 2.0:
                    break
                if _can_add_with_dp(dp_best, best_set, cand, n):
                    best_set.append(cand)
                    best_set_set.add(cand)
                    unused.remove(cand)
                    n += 1
                    improved = True
                    if n >= 48:
                        return best_set
                    break
            
            if improved:
                continue
                
            # No simple addition works. Try swaps.
            # For each element in the set, try removing it and adding a candidate
            # that reduces conflicts
            best_swap = None
            best_swap_conflicts = float('inf')
            
            # Sample a subset of elements to try removing (for speed)
            elements_to_try = best_set[:]
            random.shuffle(elements_to_try)
            
            for rem_idx, rem_elem in enumerate(elements_to_try[:20]):  # Limit to 20 removals
                if time.time() > deadline - 3.0:
                    break
                
                # Temporarily remove rem_elem
                temp_set = [x for i, x in enumerate(best_set) if x != rem_elem]
                temp_n = len(temp_set)
                temp_dp = _build_dp_from_set(temp_set)
                
                # Try adding each unused candidate
                for cand in unused[:50]:  # Limit to 50 candidates
                    if time.time() > deadline - 3.0:
                        break
                    
                    # Check if cand can be added to temp_set
                    if _can_add_with_dp(temp_dp, temp_set, cand, temp_n):
                        # This swap is valid and maintains set size
                        # Accept it
                        best_set = temp_set + [cand]
                        best_set_set = set(best_set)
                        unused.remove(cand)
                        unused.append(rem_elem)
                        improved = True
                        break
                
                if improved:
                    break
        
        # Final polish: try to add any remaining unused elements
        if time.time() < deadline - 1.0:
            best_set = sorted(best_set)
            best_set_set = set(best_set)
            dp_best = _build_dp_from_set(best_set)
            n = len(best_set)
            
            for cand in unused:
                if cand in best_set_set:
                    continue
                if time.time() > deadline - 1.0:
                    break
                if _can_add_with_dp(dp_best, best_set, cand, n):
                    best_set.append(cand)
                    best_set_set.add(cand)
                    _update_dp(dp_best, cand, n)
                    n += 1
                    if n >= 48:
                        return best_set
    
    return best_set
