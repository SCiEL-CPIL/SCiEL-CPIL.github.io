# EVOLVE-BLOCK-START
import time
import random

def _subset_sum_bitset(elements, max_sum):
    """Compute subset-sum DP using big-int bitsets.
    Returns list dp where dp[k] is a big int with bit s set iff sum s achievable with exactly k elements.
    """
    n = len(elements)
    dp = [0] * (n + 1)
    dp[0] = 1  # bit 0
    mask = (1 << (max_sum + 1)) - 1
    for x in elements:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
        # mask all
        for k in range(1, n + 1):
            dp[k] &= mask
    return dp

def _is_valid(set_vals):
    """Check if no element is the mean of any subset of >=2 other elements."""
    n = len(set_vals)
    if n < 3:
        return True
    for a in set_vals:
        others = [x for x in set_vals if x != a]
        m = len(others)
        if m < 2:
            continue
        max_target = m * a
        if max_target > 50000:
            # Too large, use a different check
            # For large a, the target k*a for k>=2 might exceed sum of others
            # Sum of others
            s = sum(others)
            if 2 * a > s:
                # Even the smallest subset (size 2) can't reach 2a if 2a > s? No, 2a could be <= s
                # Actually if 2*a > sum(others), then no subset can sum to 2a or more
                # But we need to check all k*a for k from 2 to m
                # If 2*a > s, then k*a >= 2a > s for all k>=2, so no violation
                continue
        dp = _subset_sum_bitset(others, max_target)
        for k in range(2, m + 1):
            target = k * a
            if target <= max_target:
                if (dp[k] >> target) & 1:
                    return False
    return True

def _check_addition(current_set, x):
    """Check if adding x to current_set creates any violation.
    Only need to check:
    1. x is mean of subset of current_set (size >= 2)
    2. For each a in current_set, a is mean of subset of (current_set - {a} + {x}) that includes x
    """
    n = len(current_set)
    
    # Check 1: x is mean of subset of current_set with size >= 2
    if n >= 2:
        m = n
        max_target = m * x
        if max_target <= 50000:
            dp = _subset_sum_bitset(current_set, max_target)
            for k in range(2, m + 1):
                target = k * x
                if target <= max_target and (dp[k] >> target) & 1:
                    return False
        else:
            # x is large, check if 2*x > sum(current_set)
            s = sum(current_set)
            if 2 * x <= s:
                # Need to check, but max_target too large. Use a bounded check.
                # Only check small k values
                for k in range(2, min(m, 10) + 1):
                    target = k * x
                    # Check if any subset of size k sums to target
                    # Use a smaller DP
                    dp_small = _subset_sum_bitset(current_set, target)
                    if (dp_small[k] >> target) & 1:
                        return False
    
    # Check 2: For each a in current_set, is a the mean of a subset of (current_set - {a} + {x}) that includes x?
    for a in current_set:
        others_no_a = [y for y in current_set if y != a]
        m = len(others_no_a)
        if m < 1:
            continue
        # We need: sum(T) + x = (|T|+1) * a for some T subset of others_no_a, |T| >= 0
        # => sum(T) = (|T|+1)*a - x
        # Let t = |T|, then sum(T) = (t+1)*a - x
        # t ranges from 0 to m
        max_t = m
        max_target = (max_t + 1) * a - x
        if max_target < 0:
            continue
        if max_target > 50000:
            # Check only small t
            for t in range(0, min(m, 8) + 1):
                target = (t + 1) * a - x
                if target < 0:
                    continue
                if target == 0 and t == 0:
                    # Empty subset, sum=0, check if 0 = a - x => a = x, but x not in current_set
                    continue
                dp_small = _subset_sum_bitset(others_no_a, target)
                if t <= m and (dp_small[t] >> target) & 1:
                    return False
            continue
        
        dp = _subset_sum_bitset(others_no_a, max_target)
        for t in range(0, m + 1):
            target = (t + 1) * a - x
            if 0 <= target <= max_target:
                if (dp[t] >> target) & 1:
                    return False
    
    return True

def _greedy_construction(order, deadline):
    """Greedy construction given an order of candidates."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _check_addition(selected, x):
            selected.append(x)
    return selected

def _local_search(initial_set, deadline):
    """Try to improve the set by swapping elements."""
    current = initial_set[:]
    best = current[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for x in range(1, 513):
                if x in current:
                    continue
                candidate = current[:i] + [x] + current[i+1:]
                if _is_valid(candidate):
                    # Try to extend
                    for y in range(1, 513):
                        if y in candidate:
                            continue
                        if len(candidate) >= 48:
                            break
                        if _check_addition(candidate, y):
                            candidate.append(y)
                            improved = True
                            break
                    current = candidate
                    if len(current) > len(best):
                        best = current[:]
                    break
            if improved:
                break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = []
    
    # Strategy 1: Greedy from low to high
    order = list(range(1, 513))
    selected = _greedy_construction(order, deadline)
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 2: Greedy from high to low
    order = list(range(512, 0, -1))
    selected = _greedy_construction(order, deadline)
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Strategy 3: Multiple randomized greedy
    for seed in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        selected = _greedy_construction(order, deadline)
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Strategy 4: Local search on best
    if time.time() < deadline - 10:
        improved = _local_search(best_set, deadline - 5)
        if len(improved) > len(best_set):
            best_set = improved[:]
    
    # Strategy 5: More randomized greedy with bias toward gaps
    if time.time() < deadline - 5:
        for seed in range(500, 700):
            if time.time() > deadline:
                break
            rng = random.Random(seed)
            # Bias toward numbers that create larger gaps
            order = list(range(1, 513))
            # Weight by distance from existing best elements
            weights = []
            for x in order:
                if x in best_set:
                    weights.append(0.1)
                else:
                    min_dist = min(abs(x - b) for b in best_set) if best_set else 512
                    weights.append(min_dist)
            # Sample without replacement using weights
            selected = []
            remaining = list(range(1, 513))
            rem_weights = weights[:]
            while remaining and len(selected) < 48 and time.time() < deadline:
                total_w = sum(rem_weights)
                if total_w <= 0:
                    break
                r = rng.random() * total_w
                cum = 0
                idx = len(remaining) - 1
                for j in range(len(remaining)):
                    cum += rem_weights[j]
                    if r <= cum:
                        idx = j
                        break
                x = remaining[idx]
                remaining.pop(idx)
                rem_weights.pop(idx)
                if _check_addition(selected, x):
                    selected.append(x)
            if len(selected) > len(best_set):
                best_set = selected[:]
    
    return best_set[:48]
# EVOLVE-BLOCK-END
