import time
import random

def construct():
    deadline = time.time() + 85.0
    n = 512
    target_size = 48
    
    # Precompute 3-AP conflicts for greedy ordering
    # x, y, z in AP if x+z = 2y
    # We want to avoid adding y if x, z are already present
    # Or avoid adding x if y, z are present
    
    def _build_dp(arr):
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << val)
        return dp

    def _is_valid_full(selected):
        m = len(selected)
        if m < 2:
            return True
        # Check all elements
        for i, a in enumerate(selected):
            others = [selected[j] for j in range(m) if j != i]
            if not others:
                continue
            dp_others = _build_dp(others)
            sum_others = sum(others)
            for k in range(2, len(others) + 1):
                target = k * a
                if target > sum_others:
                    break
                if (dp_others[k] >> target) & 1:
                    return False
        return True

    def _greedy_with_order(order):
        selected = []
        for x in order:
            if len(selected) >= target_size:
                break
            # Check if x can be added
            # We need to check if x is mean of subset of selected
            # AND if any element in selected is mean of subset of selected U {x} including x
            
            # For speed, we only do full check if we are close to target or at end?
            # No, we must check incrementally to ensure validity.
            
            # Check 1: x is not mean of subset of selected
            if selected:
                dp_sel = _build_dp(selected)
                sum_sel = sum(selected)
                m = len(selected)
                valid_x = True
                for k in range(2, m + 1):
                    target = k * x
                    if target > sum_sel:
                        break
                    if (dp_sel[k] >> target) & 1:
                        valid_x = False
                        break
                if not valid_x:
                    continue
                
                # Check 2: No a in selected is mean of subset of selected U {x} including x
                # Subset includes x and t elements from selected, size t+1 >= 2 => t >= 1
                # Mean a => sum(t elements) + x = (t+1)*a
                # sum(t elements) = (t+1)*a - x
                valid_a = True
                for a in selected:
                    for t in range(1, m + 1):
                        target = (t + 1) * a - x
                        if target < 0:
                            continue
                        if target > sum_sel:
                            break
                        if (dp_sel[t] >> target) & 1:
                            valid_a = False
                            break
                    if not valid_a:
                        break
                if not valid_a:
                    continue
            
            selected.append(x)
            
            # Optional: Full verification occasionally to catch bugs?
            # The incremental check is mathematically sufficient.
            
        return selected

    # Strategy 1: Greedy with numbers sorted by "conflict count" (ascending)
    # Numbers that are part of fewer 3-APs are better candidates early on?
    # Or numbers that are part of more 3-APs are worse?
    # Actually, if we pick numbers that are "isolated" in 3-APs, we are safe for k=2.
    
    conflict_count = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            # Check if i, j are part of a 3-AP with some k
            # i, j, k => 2j = i+k or 2i = j+k or 2k = i+j
            # This is O(N^3) to compute all.
            # Simpler: Just try random orders.
            pass

    best = []
    
    # Try a few deterministic orders
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        # Primes first?
        [x for x in range(1, n + 1) if x > 1 and all(x % i != 0 for i in range(2, int(x**0.5)+1))] + list(range(1, n + 1)),
        # Powers of 2 first
        [1, 2, 4, 8, 16, 32, 64, 128, 256] + list(range(1, n + 1)),
    ]
    
    for order in orders:
        if time.time() > deadline - 10.0:
            break
        res = _greedy_with_order(order)
        if len(res) > len(best):
            best = res
            if len(best) >= target_size:
                # Verify full set just in case
                if _is_valid_full(best):
                    return best
    
    # Random restarts
    for i in range(1000):
        if time.time() > deadline - 2.0:
            break
        random.seed(i)
        order = list(range(1, n + 1))
        random.shuffle(order)
        res = _greedy_with_order(order)
        if len(res) > len(best):
            best = res
            if len(best) >= target_size:
                if _is_valid_full(best):
                    return best

    return best
