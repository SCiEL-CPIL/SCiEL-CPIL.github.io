"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 87.0  # seconds, safety margin before 90s limit

def _compute_forbidden(S):
    """
    Compute the set of integers x in [1, 512] such that adding x to S
    creates a violation where x is the mean of a subset of S, or
    some a in S is the mean of a subset of S U {x}.
    """
    n = len(S)
    if n == 0:
        return set()
    
    forbidden = set()
    
    # 1. Check if x is the mean of any subset of S
    # DP for subset sums of S
    max_sum = sum(S)
    # dp[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
    # We only care about sums up to max_sum
    # Max sum is 512 * 48 = 24576. Bitset size is manageable.
    
    # To save memory/time, we can use a list of integers
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements
    
    for val in S:
        for k in range(n, 0, -1):
            if dp[k-1] != 0:
                dp[k] |= (dp[k-1] << val)
                
    for k in range(2, n + 1):
        if dp[k] == 0:
            continue
        # Iterate through set bits in dp[k]
        # We can mask to 512*n to be safe, but sum is bounded by max_sum
        bits = dp[k]
        while bits:
            # Get lowest set bit
            lsb = bits & -bits
            s = lsb.bit_length() - 1
            if s % k == 0:
                mean_val = s // k
                if 1 <= mean_val <= 512:
                    forbidden.add(mean_val)
            bits ^= lsb
            
    # 2. Check if some a in S becomes the mean of a subset containing x
    # For each a in S, we need to check if there exists a subset T of S\{a}
    # such that a is the mean of T U {x}.
    # This implies x = (|T|+1)*a - sum(T).
    # Let k = |T|+1. Then |T| = k-1.
    # x = k*a - sum(T), where T is a subset of S\{a} of size k-1.
    
    for a in S:
        S_excl_a = [v for v in S if v != a]
        n_excl = len(S_excl_a)
        if n_excl == 0:
            continue
            
        # DP for subset sums of S_excl_a
        dp_excl = [0] * (n_excl + 1)
        dp_excl[0] = 1
        
        for val in S_excl_a:
            for k in range(n_excl, 0, -1):
                if dp_excl[k-1] != 0:
                    dp_excl[k] |= (dp_excl[k-1] << val)
                    
        # For each possible subset size k (total size including x) from 2 to n_excl+1
        # T has size k-1.
        for k in range(2, n_excl + 2):
            t_size = k - 1
            if t_size > n_excl:
                break
            if dp_excl[t_size] == 0:
                continue
                
            bits = dp_excl[t_size]
            while bits:
                lsb = bits & -bits
                s = lsb.bit_length() - 1
                x_val = k * a - s
                if 1 <= x_val <= 512:
                    forbidden.add(x_val)
                bits ^= lsb
                
    return forbidden

def construct():
    start_time = time.time()
    best_set = []
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # Generate diverse orders
    orders = []
    
    # 1. Structured orders
    orders.append(candidates[:])       # Increasing
    orders.append(candidates[::-1])    # Decreasing
    
    # 2. Spread out orders
    spread_order = []
    lo, hi = 1, 512
    while lo <= hi:
        spread_order.append(hi)
        if lo != hi:
            spread_order.append(lo)
        lo += 1
        hi -= 1
    orders.append(spread_order)
    
    spread_order2 = []
    lo, hi = 1, 512
    while lo <= hi:
        spread_order2.append(lo)
        if lo != hi:
            spread_order2.append(hi)
        lo += 1
        hi -= 1
    orders.append(spread_order2)
    
    # 3. Random orders
    rng = random.Random(42)
    num_random_orders = 500 # Can afford more now
    
    for i in range(num_random_orders):
        if time.time() - start_time > _DEADLINE - 2:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
        
    # 4. Biased random orders: weight towards extremes
    for i in range(200):
        if time.time() - start_time > _DEADLINE - 2:
            break
        biased = []
        for _ in range(512):
            r = rng.random()
            if r < 0.3:
                biased.append(rng.randint(1, 50))
            elif r < 0.6:
                biased.append(rng.randint(463, 512))
            else:
                biased.append(rng.randint(1, 512))
        seen = set()
        unique_biased = []
        for v in biased:
            if v not in seen:
                seen.add(v)
                unique_biased.append(v)
        for v in range(1, 513):
            if v not in seen:
                unique_biased.append(v)
        orders.append(unique_biased)
        
    for order in orders:
        if time.time() - start_time > _DEADLINE - 1:
            break
            
        selected = []
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 1:
                break
                
            if len(selected) >= 48:
                break
                
            if x in selected:
                continue
                
            # Compute forbidden set for current selected
            # This is the expensive part, but done only when we might add x
            # Actually, we can compute it once per step of the greedy?
            # No, x changes.
            # But we can check if x is forbidden.
            # To avoid recomputing forbidden for every x, we can:
            # 1. Compute forbidden for current selected.
            # 2. Check if x is in forbidden.
            # 3. If not, add x.
            # 4. If yes, skip x.
            # Since we iterate x in order, and selected changes only when we add,
            # we can cache the forbidden set?
            # No, because we might skip many x's before adding one.
            # But if we skip x, selected doesn't change, so forbidden doesn't change.
            # So we can compute forbidden once, and reuse it for all subsequent x's
            # until we add an element.
            
            # Optimization:
            # We maintain a variable `current_forbidden` and `last_selected_size`.
            # But simpler: just compute it. It's fast.
            
            forbidden = _compute_forbidden(selected)
            
            if x not in forbidden:
                selected.append(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) >= 48:
                break
                
    # Fallback
    if not best_set:
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        
    return best_set

# EVOLVE-BLOCK-END
