import time
import random

_DEADLINE = 82.0  # safety margin before 90s limit

def _is_mean_of_subset(C, i):
    """
    Check if C[i] is the mean of any subset of at least 2 OTHER elements in C.
    Uses centered bitset DP for speed.
    Returns True if violation found.
    """
    a = C[i]
    others = C[:i] + C[i+1:]
    n = len(others)
    if n < 2:
        return False
    
    # Centered sums: z_j = y_j - a
    # We need sum(z_j) = 0 for some subset of size >= 2
    zs = [y - a for y in others]
    
    max_abs = sum(abs(z) for z in zs)
    if max_abs == 0:
        # All others equal a, impossible for distinct positive ints
        return False
        
    offset = max_abs
    mask = (1 << (2 * max_abs + 1)) - 1
    
    # dp[k]: bitset where bit s is set if sum s is achievable with exactly k elements
    # Sum 0 is at bit position offset
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
            # Early exit
            if k >= 2 and (dp[k] & (1 << offset)):
                return True
                
    for k in range(2, n + 1):
        if dp[k] & (1 << offset):
            return True
            
    return False

def _full_check(C):
    """Check if any element in C is the mean of a subset of >= 2 others."""
    for i in range(len(C)):
        if _is_mean_of_subset(C, i):
            return True
    return False

def _fast_pair_check(C, x):
    """
    Quick check for 2-element subset violations involving x.
    C is the current set (without x), x is the candidate.
    Returns True if a pair violation exists.
    """
    n_sel = len(C)
    
    # 1. x is mean of a pair in C: 2x = a + b
    for i in range(n_sel):
        for j in range(i + 1, n_sel):
            if C[i] + C[j] == 2 * x:
                return True
                
    # 2. Some a in C is mean of pair (x, b): 2a = x + b => b = 2a - x
    C_set = set(C)
    for a in C:
        b = 2 * a - x
        if b in C_set and b != a and b != x:
            return True
            
    return False

def _try_local_improvement(S, candidates, time_budget):
    """
    Try to improve S by swapping elements.
    Returns improved set if better found.
    """
    start = time.time()
    S = list(S)
    S_set = set(S)
    improved = False
    
    # Try replacing each element with a candidate not in S
    for i in range(len(S)):
        if time.time() - start > time_budget:
            break
        old_val = S[i]
        
        # Try candidates that might work
        for cand in candidates:
            if time.time() - start > time_budget:
                break
            if cand in S_set:
                continue
            
            # Quick pair check for the new element
            temp_set = S[:i] + S[i+1:]
            if _fast_pair_check(temp_set, cand):
                continue
            
            # Full check
            new_S = temp_set + [cand]
            if not _full_check(new_S):
                # Accept the swap
                S[i] = cand
                S_set.remove(old_val)
                S_set.add(cand)
                improved = True
                break  # Move to next element
                
    return S if improved else S

def construct():
    start_time = time.time()
    best_set = []
    
    candidates = list(range(1, 513))
    
    # Generate diverse orders
    orders = []
    
    # 1. Decreasing (large numbers first - less likely to be means)
    orders.append(candidates[::-1])
    
    # 2. Increasing
    orders.append(candidates[:])
    
    # 3. Interleaved high-low
    inter = []
    lo, hi = 1, 512
    while lo <= hi:
        inter.append(hi)
        if lo != hi:
            inter.append(lo)
        lo += 1
        hi -= 1
    orders.append(inter)
    
    # 4. Many random orders with different seeds
    rng = random.Random(42)
    for _ in range(200):
        if time.time() - start_time > _DEADLINE - 8:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
    
    # 5. Weighted random: bias toward extremes
    for _ in range(50):
        if time.time() - start_time > _DEADLINE - 8:
            break
        weighted = []
        for x in candidates:
            # Higher weight for extremes
            w = abs(x - 256) / 256.0 + 0.1
            weighted.extend([x] * int(w * 3))
        rng.shuffle(weighted)
        # Remove duplicates, keep order of first occurrence
        seen = set()
        unique_weighted = []
        for x in weighted:
            if x not in seen:
                seen.add(x)
                unique_weighted.append(x)
        orders.append(unique_weighted)
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        selected = []
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 3:
                break
            if len(selected) >= 48:
                break
                
            # Fast pair check
            if _fast_pair_check(selected, x):
                continue
                
            # Full check
            test_set = selected + [x]
            if _full_check(test_set):
                continue
                
            selected.append(x)
            
        if len(selected) > len(best_set):
            best_set = selected[:]
            
        if len(best_set) >= 48:
            break
    
    # Local improvement pass
    if len(best_set) < 48 and time.time() - start_time < _DEADLINE - 10:
        remaining_time = _DEADLINE - 10 - (time.time() - start_time)
        if remaining_time > 1:
            best_set = _try_local_improvement(best_set, candidates, remaining_time)
    
    # Final verification
    if best_set and _full_check(best_set):
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    elif best_set:
        return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
