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

def _is_valid_set(s):
    """Verify that no element in s is the mean of any subset of at least 2 other elements."""
    n = len(s)
    if n < 3:
        return True
    
    # Build DP table for s
    max_sum = sum(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in s:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k-1] << x)
    
    s_set = set(s)
    for a in s:
        # Check if a is mean of any subset of s \ {a} with size >= 2
        # sum(T) = k * a, |T| = k, k >= 2, T subset of s \ {a}
        # We need to check if there's a subset of s \ {a} of size k with sum k*a
        # That means: in dp[k], is there a subset of size k with sum k*a that doesn't include a?
        # Actually, we need to check subsets of s \ {a}.
        # Let's rebuild DP for s \ {a} or use a different approach.
        
        # Alternative: for each k >= 2, check if there exists a subset T of s\{a} with |T|=k and sum(T)=k*a.
        # We can check: is there a subset of s with sum k*a and size k that does NOT include a?
        # This is equivalent to: is there a subset of s\{a} with sum k*a and size k?
        
        # Efficient way: for each k, check if (k*a) is achievable as sum of k elements from s\{a}
        # We can use the fact that if a is in the subset, then we need sum of (k-1) elements from s\{a} to be k*a - a = (k-1)*a
        # So a is mean of subset T (size k, from s\{a}) iff:
        #   - There exists subset U of s\{a} with |U|=k and sum(U)=k*a
        
        # Let's just check directly using a temporary DP for s\{a}
        # But that's O(n^3) which might be slow for n=48.
        
        # Better: use the original dp and check if the bit is set AND whether a is required.
        # Actually, let's just do a simple check: for each k >= 2, target = k*a.
        # Check if dp[k] has bit target set. If yes, we need to verify that there's a subset not using a.
        # We can do this by checking if (k*a - a) is achievable with (k-1) elements from s\{a}, i.e.,
        # if there's a subset of s\{a} of size k-1 with sum (k-1)*a.
        # But that's the same problem.
        
        # Simplest correct approach: build DP for s\{a} for each a.
        # For n=48, this is 48 * 48 * 48 = ~110k operations per check, which is fine.
        
        s_without_a = [x for x in s if x != a]
        n2 = len(s_without_a)
        dp2 = [0] * (n2 + 1)
        dp2[0] = 1
        for x in s_without_a:
            for k in range(n2, 0, -1):
                dp2[k] |= (dp2[k-1] << x)
        
        for k in range(2, n2 + 1):
            target = k * a
            min_sum = k * min(s_without_a)
            max_sum = k * max(s_without_a)
            if target < min_sum or target > max_sum:
                continue
            if (dp2[k] >> target) & 1:
                return False
    
    return True

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
    for seed in range(150):
        rnd = random.Random(seed)
        rnd_order = list(range(1, 513))
        rnd.shuffle(rnd_order)
        orders.append(rnd_order)
        
    # Weighted random: bias towards edges
    for seed in range(150):
        rnd = random.Random(seed + 1000)
        nums_local = list(range(1, 513))
        keys = []
        for x in nums_local:
            dist = abs(x - 256.5)
            key = dist * 0.7 + rnd.random() * 256
            keys.append(key)
        paired = sorted(zip(keys, nums_local), key=lambda t: -t[0])
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
        dp_best = [0] * 51
        dp_best[0] = 1
        for i, x in enumerate(best_set):
            _update_dp(dp_best, x, i)
            
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            n = len(best_set)
            for cand in range(1, 513):
                if cand in best_set_set:
                    continue
                if time.time() > deadline - 2.0:
                    break
                if _can_add_with_dp(dp_best, best_set, cand, n):
                    best_set.append(cand)
                    best_set_set.add(cand)
                    _update_dp(dp_best, cand, n)
                    best_len += 1
                    improved = True
                    if best_len >= 48:
                        return best_set
                    break
    
    return best_set
