"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 85.0  # seconds, safety margin before 90s limit


def _check_x_is_mean_of_subset(x, selected):
    """
    Check if x is the mean of any subset of selected with size >= 2.
    Uses bitset DP.
    """
    n = len(selected)
    if n < 2:
        return False
    
    # We need: exists subset S of selected, |S| >= 2, such that sum(S) = |S| * x
    # Equivalently: sum(S \ {x}) ... wait, x is not in selected.
    # sum(S) = |S| * x => sum(S) - |S|*x = 0
    # Let z_i = selected[i] - x. We need sum of z over subset = 0, |subset| >= 2.
    
    zs = [y - x for y in selected]
    
    # Compute bounds
    pos_sum = sum(z for z in zs if z > 0)
    neg_sum = sum(-z for z in zs if z < 0)
    offset = pos_sum  # or max(pos_sum, neg_sum), but we need to cover both
    
    # Actually, the sum can range from neg_sum to pos_sum
    # offset should be such that sum + offset >= 0
    # min possible sum = neg_sum, max = pos_sum
    # So offset = -neg_sum = sum of abs of negative parts... 
    # Let's use offset = pos_sum + neg_sum / 2? No.
    # offset = -min_sum = neg_sum
    offset = neg_sum
    total_range = pos_sum + neg_sum
    
    if total_range == 0:
        # All z are 0, so any subset sums to 0
        return n >= 2
    
    mask = (1 << (total_range + 1)) - 1
    
    # dp[k] = bitset of achievable sums (shifted by offset) with exactly k elements
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
    
    # Check if sum 0 (i.e., bit at offset) is achievable with k >= 2
    target_bit = 1 << offset
    for k in range(2, n + 1):
        if dp[k] & target_bit:
            return True
    return False


def _check_existing_violation(x, selected):
    """
    Check if any element a in selected becomes the mean of a subset that includes x.
    For each a in selected, we need: exists subset U of (selected \ {a}) with |U| >= 1
    such that sum(U) + x = (|U| + 1) * a
    => sum(U) - |U|*a = a - x
    Let w_i = selected[i] - a for i != a. We need sum of w over U = a - x, |U| >= 1.
    """
    n = len(selected)
    if n < 1:
        return False
    
    for idx, a in enumerate(selected):
        # Others = selected without a
        others = selected[:idx] + selected[idx+1:]
        m = len(others)
        if m < 1:
            continue
        
        target = a - x
        
        # We need subset U of others with sum(w) = target, where w_i = others[i] - a
        # |U| >= 1
        ws = [y - a for y in others]
        
        pos_sum = sum(w for w in ws if w > 0)
        neg_sum = sum(-w for w in ws if w < 0)
        total_range = pos_sum + neg_sum
        
        if total_range == 0:
            # All w are 0, so any subset sums to 0
            if target == 0 and m >= 1:
                return True
            continue
        
        offset = neg_sum
        mask = (1 << (total_range + 1)) - 1
        
        # dp[k] = bitset of achievable sums with exactly k elements
        dp = [0] * (m + 1)
        dp[0] = 1 << offset
        
        for w in ws:
            for k in range(m, 0, -1):
                if dp[k-1] == 0:
                    continue
                if w >= 0:
                    shifted = dp[k-1] << w
                else:
                    shifted = dp[k-1] >> (-w)
                dp[k] |= (shifted & mask)
        
        target_bit = 1 << (offset + target)
        for k in range(1, m + 1):
            if dp[k] & target_bit:
                return True
    
    return False


def _quick_pair_check(x, selected):
    """
    Quick O(n) check for 2-element subset violations.
    1. Is x the mean of a pair in selected? i.e., exists i<j: selected[i]+selected[j]=2x
    2. Is any a in selected the mean of pair (x, y) for some y in selected, y != a?
       i.e., 2a = x + y => y = 2a - x, and y in selected, y != a
    """
    n = len(selected)
    
    # Check 1: x is mean of pair in selected
    # For each i, check if 2x - selected[i] is in selected and != selected[i]
    sel_set = set(selected)
    for i in range(n):
        complement = 2 * x - selected[i]
        if complement in sel_set and complement != selected[i]:
            return True
    
    # Check 2: any a in selected is mean of (x, y)
    for a in selected:
        y = 2 * a - x
        if y in sel_set and y != a:
            return True
    
    return False


def _full_check(C):
    """
    Full verification: check if any element in C is the mean of a subset of at least 2 others.
    """
    for i in range(len(C)):
        a = C[i]
        others = C[:i] + C[i+1:]
        n = len(others)
        if n < 2:
            continue
        
        zs = [y - a for y in others]
        pos_sum = sum(z for z in zs if z > 0)
        neg_sum = sum(-z for z in zs if z < 0)
        total_range = pos_sum + neg_sum
        
        if total_range == 0:
            if n >= 2:
                return True
            continue
        
        offset = neg_sum
        mask = (1 << (total_range + 1)) - 1
        
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
        
        target_bit = 1 << offset
        for k in range(2, n + 1):
            if dp[k] & target_bit:
                return True
    
    return False


def _greedy_build(order, deadline):
    """
    Greedily build a valid set from the given order.
    Returns the constructed set.
    """
    selected = []
    
    for x in order:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        
        # Quick pair check
        if _quick_pair_check(x, selected):
            continue
        
        # Check if x is mean of subset of selected (size >= 2)
        if len(selected) >= 2:
            if _check_x_is_mean_of_subset(x, selected):
                continue
        
        # Check if any existing element becomes mean of subset including x
        if len(selected) >= 1:
            if _check_existing_violation(x, selected):
                continue
        
        selected.append(x)
    
    return selected


def construct():
    start_time = time.time()
    best_set = []
    
    candidates = list(range(1, 513))
    
    # Generate multiple candidate orders
    orders = []
    
    # 1. Extremes first (spread out)
    lo, hi = 0, len(candidates) - 1
    spread_order = []
    while lo <= hi:
        spread_order.append(candidates[hi])
        if lo != hi:
            spread_order.append(candidates[lo])
        lo += 1
        hi -= 1
    orders.append(spread_order)
    
    # 2. Powers of 2 first, then others
    pows = [2**k for k in range(9) if 2**k <= 512]
    rest = [x for x in candidates if x not in pows]
    orders.append(pows + rest)
    
    # 3. Odds then evens
    orders.append([x for x in candidates if x % 2 != 0] + [x for x in candidates if x % 2 == 0])
    
    # 4. Descending
    orders.append(sorted(candidates, reverse=True))
    
    # 5. Ascending
    orders.append(candidates[:])
    
    # 6. Random orders
    rng = random.Random(42)
    for _ in range(20):
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
    
    # 7. Structured: multiples of large numbers first
    for base in [10, 7, 5, 4, 3]:
        multiples = [x for x in candidates if x % base == 0]
        non_mult = [x for x in candidates if x % base != 0]
        orders.append(multiples + non_mult)
    
    deadline = start_time + _DEADLINE - 5
    
    for order in orders:
        if time.time() > deadline:
            break
        
        result = _greedy_build(order, deadline)
        
        if len(result) > len(best_set):
            best_set = result[:]
            
            # If we hit 48, we're done
            if len(best_set) >= 48:
                break
    
    # Final verification
    if best_set:
        if _full_check(best_set):
            # Fallback to a known valid set
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
