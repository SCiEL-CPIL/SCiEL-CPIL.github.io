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
        if (dp[k] >> target) & 1:
            return False
            
    # 2. No a in current_set is mean of subset of (current_set U {cand}) including cand.
    # For each a in current_set, check if there exists T subset of current_set, |T|>=1
    # such that sum(T) + cand = (|T|+1) * a
    # i.e., sum(T) = a*(|T|+1) - cand
    for a in current_set:
        for k in range(1, n + 1):
            target = a * (k + 1) - cand
            if 0 <= target and (dp[k] >> target) & 1:
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
    for seed in range(20):
        rnd = random.Random(seed)
        rnd_order = list(range(1, 513))
        rnd.shuffle(rnd_order)
        orders.append(rnd_order)

    for order in orders:
        if time.time() > deadline - 10.0:
            break
            
        current_set_list = []
        max_dp_size = 50
        dp = [0] * (max_dp_size + 1)
        dp[0] = 1
        
        for cand in order:
            if time.time() > deadline - 5.0:
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

    # Strategy 2: Local search - try to improve best_set by swapping elements
    if best_len > 0 and time.time() < deadline - 10.0:
        current_set = best_set[:]
        current_set_set = set(current_set)
        
        # Rebuild DP for current_set
        max_dp_size = 50
        dp = [0] * (max_dp_size + 1)
        dp[0] = 1
        for elem in current_set:
            n = len([x for x in current_set if x < elem])  # This is wrong, need proper count
            # Actually, let's rebuild properly
        dp = [0] * (max_dp_size + 1)
        dp[0] = 1
        for elem in current_set:
            n_prev = len(current_set) - 1  # This is also wrong
            # Let me just rebuild from scratch
        dp = [0] * (max_dp_size + 1)
        dp[0] = 1
        for elem in current_set:
            # Find index of elem in sorted position... this is getting complicated
            # Just do it properly:
            pass
        
        # Simpler: rebuild DP by processing elements in order
        dp = [0] * (max_dp_size + 1)
        dp[0] = 1
        for elem in current_set:
            n_prev = len([e for e in current_set if e != elem])  # still wrong
            # OK let me just track size properly
        dp = [0] * (max_dp_size + 1)
        dp[0] = 1
        cnt = 0
        for elem in current_set:
            _update_dp(dp, elem, cnt)
            cnt += 1
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    a = current_set[i]
                    b = current_set[j]
                    # Try removing a and adding b, or removing b and adding a
                    # Actually, try swapping: remove a, add some other number
                    pass
        
        # Try: for each element in current_set, try removing it and adding a different number
        for i in range(len(current_set)):
            if time.time() > deadline - 5.0:
                break
            elem_to_remove = current_set[i]
            remaining = [x for x in current_set if x != elem_to_remove]
            remaining_set = set(remaining)
            
            # Rebuild DP for remaining
            dp_rem = [0] * (max_dp_size + 1)
            dp_rem[0] = 1
            cnt = 0
            for elem in remaining:
                _update_dp(dp_rem, elem, cnt)
                cnt += 1
            
            # Try adding each number not in remaining
            for cand in nums:
                if cand in remaining_set:
                    continue
                if _can_add_with_dp(dp_rem, remaining, cand, len(remaining)):
                    new_set = remaining + [cand]
                    if len(new_set) > len(current_set):
                        current_set = new_set
                        improved = True
                        break
                if time.time() > deadline - 5.0:
                    break
            if improved:
                break
    
    if len(current_set) > best_len:
        best_set = current_set
        best_len = len(best_set)
    
    return best_set
