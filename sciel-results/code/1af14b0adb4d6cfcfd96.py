"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit

    best_set = []
    best_len = 0

    # Helper to check if x is a mean of a subset of size k
    # dp[k] is a bitmask of achievable sums with exactly k elements
    def is_mean_of_subset(dp, x, k):
        target = k * x
        if target < 0 or target > 24576:
            return False
        return (dp[k] >> target) & 1

    def try_greedy_v2(order):
        selected = []
        n = 0
        dp_all = [1] # dp_all[0] = 1
        dp_without = {}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= 48:
                break
            
            valid = True
            
            # Check 1
            for k in range(2, n + 1):
                target = k * x
                if target > 24576:
                    break
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2
            for a in selected:
                dp_w = dp_without[a]
                for k in range(1, n):
                    target = (k + 1) * a - x
                    if target < 0 or target > 24576:
                        continue
                    if (dp_w[k] >> target) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Valid, add x
            selected.append(x)
            
            # Save old state for x's exclusion set
            old_dp_all = dp_all
            
            # Update dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = old_dp_all[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (old_dp_all[k-1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without for existing elements
            new_dp_without = {}
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k-1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For x, the set of others is the old selected set.
            # So dp_without[x] is old_dp_all.
            # We need to pad old_dp_all to size n+2 (new n is old_n + 1)
            # old_dp_all had size n+1 (indices 0..n).
            # new_dp_without[x] needs size (n+1)+1 = n+2?
            # The list size should be consistent with max k.
            # Max k for dp_without[x] is n (since there are n other elements).
            # So we need indices 0..n.
            # old_dp_all has indices 0..n.
            # So we can just use a copy.
            new_dp_without[x] = old_dp_all + [0] * max(0, (n + 1) - len(old_dp_all))
            
            dp_without = new_dp_without
            n += 1
            
        return selected

    # Strategies
    orders = []
    
    # 1. Spread
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(lo)
        if lo != hi:
            spread.append(hi)
        lo += 1
        hi -= 1
    orders.append(spread)
    
    # 2. Random shuffles
    for i in range(20):
        random.seed(i)
        cand = list(range(1, 513))
        random.shuffle(cand)
        orders.append(cand)
        
    # 3. Ascending/Descending
    orders.append(list(range(1, 513)))
    orders.append(list(range(512, 0, -1)))
    
    for order in orders:
        if time.time() - start_time > deadline:
            break
        res = try_greedy_v2(order)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break
                
    return best_set

# EVOLVE-BLOCK-END
