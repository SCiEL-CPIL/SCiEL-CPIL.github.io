"""
Cardinality-separated subset sums: Optimized incremental bitset-DP with refined local search.
"""
import time
import random

def _compute_dp(arr):
    """
    Compute cardinality-indexed bitset DP for arr.
    dp[k] is a bitmask where bit s is set iff sum s is achievable with exactly k elements.
    """
    n = len(arr)
    dp = [0] * (n + 1)
    dp[0] = 1  # Empty subset: sum 0
    
    for x in arr:
        # Iterate backwards to avoid using the same element multiple times in one step
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    return dp

def _is_valid_dp(dp):
    """
    Check if no sum appears in two different cardinalities.
    dp[k] & seen must be 0 for all k, where seen is the union of all dp[j] for j < k.
    """
    seen = 0
    for k in range(len(dp)):
        if dp[k] & seen:
            return False
        seen |= dp[k]
    return True

def _try_add_fast(arr, x, dp_cache=None):
    """
    Try to add element x to arr.
    Returns (True, new_dp) if valid, else (False, None).
    Uses dp_cache if provided and matches len(arr).
    """
    n = len(arr)
    
    # Initialize dp
    if dp_cache is not None and len(dp_cache) == n + 1:
        dp = list(dp_cache)
    else:
        dp = _compute_dp(arr)
        
    # Create new_dp for n+1 elements
    new_dp = dp + [0]
    
    # Add x
    # For k from n+1 down to 1: new_dp[k] |= old_dp[k-1] << x
    # Note: new_dp[k] initially holds old_dp[k] (from the list copy/extension)
    # We must be careful: new_dp[k] = old_dp[k] | (old_dp[k-1] << x)
    # But we are iterating downwards, so old_dp[k-1] is not yet modified in new_dp?
    # Actually, new_dp is a new list. new_dp[k] starts as dp[k] (except new_dp[n+1] is 0).
    # We update new_dp[k] |= dp[k-1] << x.
    # Since we iterate k from n+1 down to 1, and we read from dp (the old array), this is safe.
    
    for k in range(n + 1, 0, -1):
        new_dp[k] |= (dp[k - 1] << x)
        
    if not _is_valid_dp(new_dp):
        return False, None
        
    return True, new_dp

def _grow(cur, order, deadline):
    """
    Greedy growth: try to add elements from 'order' to 'cur' if valid.
    Repeats until no more can be added or deadline is reached.
    """
    cur = list(cur)
    cur_set = set(cur)
    dp = _compute_dp(cur)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in order:
            if time.time() > deadline:
                break
            if x in cur_set:
                continue
            
            ok, new_dp = _try_add_fast(cur, x, dp)
            if ok:
                cur.append(x)
                cur_set.add(x)
                dp = new_dp
                improved = True
                
    return cur

def _local_search(cur, deadline):
    """
    Local search to improve set size.
    Neighborhoods:
    1. 1-for-2: Remove one, add two (net +1)
    2. 2-for-3: Remove two, add three (net +1)
    3. 1-for-1: Swap one for another (net 0, but might enable future growth)
    """
    cur = list(cur)
    n_max = 256
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(cur)
        
        # --- Neighborhood 1: 1-for-2 ---
        # Remove one element, try to add two new ones
        for i in range(n):
            if time.time() > deadline:
                return cur
            
            x_remove = cur[i]
            base = cur[:i] + cur[i+1:]
            base_set = set(base)
            base_dp = _compute_dp(base)
            
            if not _is_valid_dp(base_dp):
                continue
            
            # Try to add 'a'
            for a in range(1, n_max + 1):
                if time.time() > deadline:
                    return cur
                if a == x_remove or a in base_set:
                    continue
                
                ok1, dp1 = _try_add_fast(base, a, base_dp)
                if not ok1:
                    continue
                
                base2 = base + [a]
                base2_set = base_set | {a}
                
                # Try to add 'b' (b > a to avoid duplicates/permutations)
                for b in range(a + 1, n_max + 1):
                    if time.time() > deadline:
                        return cur
                    if b == x_remove or b in base2_set:
                        continue
                    
                    ok2, _ = _try_add_fast(base2, b, dp1)
                    if ok2:
                        cur = base2 + [b]
                        improved = True
                        break
                    if time.time() > deadline:
                        break
                if improved or time.time() > deadline:
                    break
            if improved or time.time() > deadline:
                break
        
        # --- Neighborhood 2: 2-for-3 ---
        # Remove two, add three
        if not improved and time.time() < deadline:
            n = len(cur)
            for i in range(n):
                if time.time() > deadline:
                    return cur
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return cur
                    
                    x1, x2 = cur[i], cur[j]
                    base = cur[:i] + cur[i+1:j] + cur[j+1:]
                    base_set = set(base)
                    base_dp = _compute_dp(base)
                    
                    if not _is_valid_dp(base_dp):
                        continue
                    
                    for a in range(1, n_max + 1):
                        if time.time() > deadline:
                            return cur
                        if a == x1 or a == x2 or a in base_set:
                            continue
                        
                        ok1, dp1 = _try_add_fast(base, a, base_dp)
                        if not ok1:
                            continue
                        
                        base2 = base + [a]
                        base2_set = base_set | {a}
                        
                        for b in range(a + 1, n_max + 1):
                            if time.time() > deadline:
                                return cur
                            if b == x1 or b == x2 or b in base2_set:
                                continue
                            
                            ok2, dp2 = _try_add_fast(base2, b, dp1)
                            if not ok2:
                                continue
                            
                            base3 = base2 + [b]
                            base3_set = base2_set | {b}
                            
                            for c in range(b + 1, n_max + 1):
                                if time.time() > deadline:
                                    return cur
                                if c == x1 or c == x2 or c in base3_set:
                                    continue
                                
                                ok3, _ = _try_add_fast(base3, c, dp2)
                                if ok3:
                                    cur = base3 + [c]
                                    improved = True
                                    break
                                if time.time() > deadline:
                                    break
                            if improved or time.time() > deadline:
                                break
                        if improved or time.time() > deadline:
                            break
                    if improved or time.time() > deadline:
                        break
                if improved or time.time() > deadline:
                    break

        # --- Neighborhood 3: 1-for-1 Swap ---
        # This is useful to escape local optima. We accept a swap if it keeps validity.
        # To avoid infinite loops or useless swaps, we only accept if it changes the set 
        # and we track visited states (simplified: just accept first valid different swap).
        if not improved and time.time() < deadline:
            n = len(cur)
            for i in range(n):
                if time.time() > deadline:
                    return cur
                
                x = cur[i]
                base = cur[:i] + cur[i+1:]
                base_set = set(base)
                base_dp = _compute_dp(base)
                
                if not _is_valid_dp(base_dp):
                    continue
                
                # Prefer larger swaps? Or random?
                # Let's try to find a swap that allows more small numbers later.
                # Heuristic: Replace x with a smaller number if possible? 
                # Or replace with a number that 'unblocks' something.
                # Simple approach: Try all candidates.
                for a in range(1, n_max + 1):
                    if time.time() > deadline:
                        return cur
                    if a == x or a in base_set:
                        continue
                    
                    ok, _ = _try_add_fast(base, a, base_dp)
                    if ok:
                        # Accept the swap
                        cur = base + [a]
                        improved = True
                        break
                if improved:
                    break

    return cur

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s. Return by 85s.
    deadline = start_time + 85.0
    rng = random.Random(42)
    
    best = []
    best_len = 0
    
    # Candidate orders
    candidates_all = list(range(1, 257))
    orders = [
        candidates_all,
        list(reversed(candidates_all)),
        # Alternating small/large
        [i for i in range(1, 257, 2)] + [i for i in range(2, 257, 2)],
        [i for i in range(2, 257, 2)] + [i for i in range(1, 257, 2)],
    ]
    
    # Add a few shuffled orders for diversity
    for _ in range(5):
        o = list(candidates_all)
        rng.shuffle(o)
        orders.append(o)
        
    # Base sets
    bases = [
        [],
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 3, 7, 15, 31, 63, 127],
        [2, 5, 11, 23, 47, 95],
    ]

    # Phase 1: Greedy Growth from multiple starts
    for base in bases:
        if time.time() > deadline - 15:
            break
            
        # Validate base
        if base:
            if not _is_valid_dp(_compute_dp(base)):
                continue
                
        for order in orders:
            if time.time() > deadline - 15:
                break
                
            cur = _grow(list(base), order, deadline - 15)
            if len(cur) > best_len:
                best = list(cur)
                best_len = len(cur)
                
    # Phase 2: Local Search to improve best
    if best and time.time() < deadline - 5:
        # Run local search for a while
        while time.time() < deadline - 5:
            before_len = len(best)
            best = _local_search(best, deadline - 5)
            if len(best) <= before_len:
                break
            best_len = len(best)
            
    # Phase 3: Final Greedy Push
    if best and time.time() < deadline - 2:
        # Try to add any remaining candidates one last time
        best = _grow(best, candidates_all, deadline - 2)
        best_len = len(best)
        
    # Phase 4: Last chance local search
    if best and time.time() < deadline - 1:
        best = _local_search(best, deadline - 0.5)
        best_len = len(best)

    # Ensure output is a list of Python ints
    return [int(v) for v in best]
