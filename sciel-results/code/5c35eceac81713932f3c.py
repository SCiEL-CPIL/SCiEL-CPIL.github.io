import time
import random

def _fast_check(S, x):
    """
    Returns True if x can be added to S without creating a mean violation 
    with subset size 2 or 3.
    S is a list of ints. x is an int not in S.
    """
    S_set = set(S)
    n = len(S)
    
    # 1. Check subset size 2
    # x = (a+b)/2 => 2x = a+b
    for a in S:
        b = 2 * x - a
        if b in S_set and b != a:
            return False
            
    # y = (x+b)/2 => 2y = x+b => b = 2y-x
    for y in S:
        b = 2 * y - x
        if b in S_set and b != y and b != x:
            return False

    # 2. Check subset size 3
    # x = (a+b+c)/3 => 3x = a+b+c
    # Precompute pair sums of S? 
    # Since n is small (< 48), O(n^2) is fine.
    # We need distinct a, b, c.
    
    # Check if x is mean of 3 elements in S
    # 3x = a + b + c
    # Iterate a, b. c = 3x - a - b.
    # Ensure c in S, c distinct from a, b.
    # To avoid duplicates and ensure distinctness efficiently:
    # We can just iterate.
    
    # Optimization: Sort S? No, S is list.
    # Just iterate.
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            c = 3 * x - a - b
            if c in S_set and c != a and c != b:
                # We must ensure c is not a or b.
                # Also, we must ensure a, b, c are distinct.
                # The check c != a and c != b handles it.
                # But we also need to ensure that the pair (a,b) we picked 
                # doesn't accidentally use c as a or b? 
                # No, a and b are fixed from indices i, j.
                # c is calculated.
                # We just need to make sure c is in S.
                return False

    # Check if y in S is mean of {x, a, b} with a,b in S
    # 3y = x + a + b => a + b = 3y - x
    # Iterate y in S.
    # For each y, we need to find a, b in S \ {y} such that a+b = 3y-x.
    # And a, b distinct from each other and y.
    
    # We can reuse the pair sum idea.
    # Let target_sum_y = 3y - x.
    # We need a pair (a,b) in S such that a+b = target_sum_y, a!=b, a!=y, b!=y.
    
    # Since n is small, we can just do O(n^2) for each y? 
    # Total O(n^3) ~ 48^3 = 110k. 
    # 512 candidates * 110k = 56M ops. 
    # This might be too slow for Python if done naively in a tight loop.
    
    # Better: Precompute all pair sums of S once per call?
    # But S changes? No, S is fixed for this check.
    # We can build a dictionary: sum -> list of (a,b) pairs?
    # Or just set of sums? 
    # If sum is in set, we need to verify distinctness.
    
    # Let's optimize the size-3 check for y.
    # Build a set of all pair sums of S?
    # pair_sums = set()
    # for i in range(n):
    #   for j in range(i+1, n):
    #     pair_sums.add(S[i] + S[j])
    
    # Then for each y:
    #   target = 3*y - x
    #   if target in pair_sums:
    #      # Need to verify that there exists a pair (a,b) with a+b=target
    #      # such that a!=y, b!=y, a!=b.
    #      # This verification is O(n) if we store the pairs, or O(n^2) if we re-search.
    #      # Storing pairs: dict sum -> list of pairs.
    #      # Memory is fine.
    
    # Let's do this.
    pair_sums_dict = {}
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            s = a + b
            if s not in pair_sums_dict:
                pair_sums_dict[s] = []
            pair_sums_dict[s].append((a, b))
    
    for y in S:
        target = 3 * y - x
        if target in pair_sums_dict:
            pairs = pair_sums_dict[target]
            for a, b in pairs:
                if a != y and b != y:
                    # a and b are distinct by construction (i != j)
                    # a, b != y checked
                    # x is distinct from S
                    return False
            # If no valid pair found, continue to next y

    return True

def _is_valid_set(S):
    """
    Full DP check.
    """
    if len(S) < 3:
        return True
        
    S = list(S)
    N = len(S)
    
    for i in range(N):
        a = S[i]
        others = S[:i] + S[i+1:]
        n_others = len(others)
        
        if n_others < 2:
            continue
            
        max_sum = sum(others)
        
        dp = [0] * (n_others + 1)
        dp[0] = 1
        
        for val in others:
            for k in range(n_others - 1, -1, -1):
                if dp[k] == 0:
                    continue
                dp[k+1] |= dp[k] << val
                
        for k in range(2, n_others + 1):
            if dp[k] == 0:
                continue
            target_sum = k * a
            if target_sum <= max_sum:
                if (dp[k] >> target_sum) & 1:
                    return False
            else:
                # Since others are positive, sum increases with k?
                # Not necessarily, but max_sum is fixed.
                # If k*a > max_sum, it's impossible.
                pass
                
    return True

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_set = []
    candidates = list(range(1, 513))
    
    orders = []
    orders.append(candidates[::-1])
    orders.append(candidates[:])
    
    inter = []
    lo, hi = 1, 512
    while lo <= hi:
        inter.append(hi)
        if lo != hi:
            inter.append(lo)
        lo += 1
        hi -= 1
    orders.append(inter)
    
    rng = random.Random(42)
    
    # Try deterministic orders
    for order in orders:
        if time.time() - start_time > deadline - 5:
            break
        current_set = []
        for x in order:
            if time.time() - start_time > deadline - 2:
                break
            if len(current_set) >= 48:
                break
            
            if _fast_check(current_set, x):
                test_set = current_set + [x]
                if _is_valid_set(test_set):
                    current_set.append(x)
                    
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if len(best_set) >= 48:
            break
            
    # Random restarts
    while time.time() - start_time < deadline - 2:
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        
        current_set = []
        for x in rand_order:
            if time.time() - start_time > deadline - 2:
                break
            if len(current_set) >= 48:
                break
                
            if _fast_check(current_set, x):
                test_set = current_set + [x]
                if _is_valid_set(test_set):
                    current_set.append(x)
                    
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            if len(best_set) >= 48:
                break
                
    if _is_valid_set(best_set):
        return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
