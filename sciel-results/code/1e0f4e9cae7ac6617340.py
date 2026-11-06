import time
import random

def _can_add_with_dp(dp, current_set, cand, n):
    """
    Check if cand can be added to current_set given the DP table for current_set.
    Returns True if safe. n = len(current_set)
    """
    if n == 0:
        return True
    
    # 1. cand is not mean of any subset of current_set (size >= 2)
    for k in range(2, n + 1):
        target = k * cand
        if target > 24576:  # max possible sum is 48*512=24576
            break
        if (dp[k] >> target) & 1:
            return False
            
    # 2. No a in current_set is mean of subset of (current_set U {cand}) including cand.
    # For each a in current_set, check if there exists T subset of current_set, |T|>=1
    # such that sum(T) + cand = (|T|+1) * a
    # i.e., sum(T) = a*(|T|+1) - cand
    # Optimize: skip if target out of range
    for a in current_set:
        # target = a*(k+1) - cand must be in [min_k_sum, max_k_sum]
        # min sum of k elements >= k*min_S, max sum of k elements <= k*max_S
        # We can early-exit if for all k, target is out of [0, 24576]
        for k in range(1, n + 1):
            target = a * (k + 1) - cand
            if target < 0 or target > 24576:
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

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_len = 0
    
    nums = list(range(1, 513))
    
    # Strategy 1: Greedy with multiple orderings
    # Edge-first ordering tends to work well
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
    # Increase number of restarts since local search is removed
    for seed in range(150):
        rnd = random.Random(seed)
        rnd_order = list(range(1, 513))
        rnd.shuffle(rnd_order)
        orders.append(rnd_order)
        
    # Weighted random: bias towards edges
    for seed in range(150):
        rnd = random.Random(seed + 1000)
        # Create a permutation biased toward edges
        # Assign random priorities, then sort by a key that favors edges
        nums = list(range(1, 513))
        # Use a random key that adds noise but biases toward extremes
        keys = []
        for x in nums:
            # Distance from center: larger means more extreme
            dist = abs(x - 256.5)
            # Random perturbation to create diversity
            key = dist * 0.7 + rnd.random() * 256
            keys.append(key)
        # Sort by key descending (extremes first)
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

    # Final: try to improve best_set with local swaps
    if time.time() < deadline - 2.0 and best_len < 48:
        best_set_set = set(best_set)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try adding each unused number
            for cand in range(1, 513):
                if cand in best_set_set:
                    continue
                if time.time() > deadline - 2.0:
                    break
                # Rebuild DP for current best_set
                n = len(best_set)
                dp_tmp = [0] * 51
                dp_tmp[0] = 1
                for x in best_set:
                    for k in range(n, 0, -1):
                        dp_tmp[k] |= (dp_tmp[k-1] << x)
                if _can_add_with_dp(dp_tmp, best_set, cand, n):
                    best_set.append(cand)
                    best_set_set.add(cand)
                    best_len += 1
                    improved = True
                    if best_len >= 48:
                        return best_set
                    break  # restart the for loop with updated set
    
    return best_set
