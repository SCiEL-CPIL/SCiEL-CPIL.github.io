"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 87.0  # seconds, safety margin before 90s limit

def _check_x_is_mean_of_subset(x, S):
    """
    Check if x is the mean of any subset of at least 2 elements in S.
    Uses bitset DP on differences.
    """
    n = len(S)
    if n < 2:
        return False
    
    zs = [y - x for y in S]
    
    # Calculate max absolute sum for offset
    max_sum = sum(abs(z) for z in zs)
    if max_sum == 0:
        return False  # All equal to x, impossible with distinct elements
        
    offset = max_sum
    mask = (1 << (2 * max_sum + 1)) - 1
    
    # dp[k] is a bitset where bit s is set if sum s-offset is achievable with exactly k elements
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    for z in zs:
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z >= 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            
            dp[k] |= (shifted & mask)
            
            # Early exit: if sum 0 is achievable with k >= 2 elements
            if k >= 2 and (dp[k] & (1 << offset)):
                return True
                
    return False

def _check_existing_violation(x, S):
    """
    Check if adding x to S creates any violation where an element a in S
    is the mean of a subset containing x and at least one other element from S.
    """
    n = len(S)
    if n < 2:
        return False
    
    for a in S:
        U = [t for t in S if t != a]
        if len(U) < 1:
            continue
        
        target_val = a - x
        
        ws = [t - a for t in U]
        max_abs = sum(abs(w) for w in ws)
        if max_abs == 0:
            continue
        
        offset = max_abs
        mask = (1 << (2 * max_abs + 1)) - 1
        
        dp = [0] * (len(U) + 1)
        dp[0] = 1 << offset
        
        for w in ws:
            for k in range(len(U), 0, -1):
                if dp[k-1] == 0:
                    continue
                if w >= 0:
                    shifted = dp[k-1] << w
                else:
                    shifted = dp[k-1] >> (-w)
                dp[k] |= (shifted & mask)
        
        if 0 <= offset + target_val <= 2 * max_abs:
            for k in range(1, len(U) + 1):
                if dp[k] & (1 << (offset + target_val)):
                    return True
    
    return False

def _quick_pair_check(x, S):
    """
    Quick check for 2-element subsets involving x.
    """
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            if S[i] + S[j] == 2 * x:
                return True
    
    S_set = set(S)
    for a in S:
        b = 2 * a - x
        if b in S_set and b != a:
            return True
    
    return False

def _full_check(C):
    """
    Full verification: check if any element in C is the mean of a subset of 
    at least 2 other elements.
    """
    for i in range(len(C)):
        a = C[i]
        others = C[:i] + C[i+1:]
        if _check_x_is_mean_of_subset(a, others):
            return True
    return False

def _generate_sidon_seed():
    """
    Generate a Sidon sequence (B2 sequence) in [1, 512].
    A Sidon sequence has all pairwise sums distinct.
    We then filter it to remove any pair-mean violations.
    """
    # Greedy Sidon construction
    S = []
    sums = set()
    for x in range(1, 513):
        valid = True
        for a in S:
            s = a + x
            if s in sums:
                valid = False
                break
            sums.add(s)
        if valid:
            S.append(x)
    
    # Filter S to ensure no element is the mean of a pair of others
    # This is a "weakly sum-free" check for pairs
    valid_S = []
    S_set = set(S)
    for x in S:
        # Check if x is mean of a, b in S \ {x}
        # 2x = a + b
        is_mean = False
        for a in S:
            if a == x:
                continue
            b = 2 * x - a
            if b in S_set and b != x and b > a: # b > a to avoid double check? No, just check existence
                # Need to ensure b is in S and b != x
                if b in S_set and b != x:
                    is_mean = True
                    break
        if not is_mean:
            valid_S.append(x)
            
    # Also check if any a in valid_S is mean of pair in valid_S \ {a}
    # This is harder to do incrementally, so we just return valid_S and let the main loop refine
    # Actually, the main loop checks for violations.
    return valid_S

def construct():
    start_time = time.time()
    best_set = []
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # Generate diverse orders
    orders = []
    
    # 1. Sidon-based seed order
    # We want to prioritize elements that form a good Sidon set
    sidon_seed = _generate_sidon_seed()
    # Create an order that puts sidon_seed first, then the rest
    # We shuffle the sidon_seed a bit to allow different variations?
    # Or just use it as a priority list.
    rest = [x for x in candidates if x not in sidon_seed]
    orders.append(sidon_seed + rest)
    
    # 2. Structured orders
    orders.append(candidates[:])       # Increasing
    orders.append(candidates[::-1])    # Decreasing
    
    # 3. Spread out orders
    spread_order = []
    lo, hi = 1, 512
    while lo <= hi:
        spread_order.append(hi)
        if lo != hi:
            spread_order.append(lo)
        lo += 1
        hi -= 1
    orders.append(spread_order)
    
    # 4. Random orders with different seeds
    rng = random.Random(42)
    num_random_orders = 300
    
    for i in range(num_random_orders):
        if time.time() - start_time > _DEADLINE - 3:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
    
    # 5. Biased random orders: weight towards extremes
    for i in range(100):
        if time.time() - start_time > _DEADLINE - 3:
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
        if time.time() - start_time > _DEADLINE - 2:
            break
            
        selected = []
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 2:
                break
            
            if len(selected) >= 48:
                break
                
            if _quick_pair_check(x, selected):
                continue
            
            if _check_x_is_mean_of_subset(x, selected):
                continue
            
            if _check_existing_violation(x, selected):
                continue
            
            selected.append(x)
            
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) >= 48:
                break
    
    if best_set:
        if not _full_check(best_set):
            return best_set
    
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
