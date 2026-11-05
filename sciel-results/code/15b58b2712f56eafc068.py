import time
import random
import sys
from itertools import combinations

_DEADLINE = 85.0

def _check_violation(C, idx):
    """
    Check if C[idx] is the mean of any subset of >=2 OTHER elements.
    Uses bitset DP.
    """
    a = C[idx]
    n = len(C)
    if n < 3:
        return False
    
    diffs = [C[j] - a for j in range(n) if j != idx]
    m = len(diffs)
    if m < 2:
        return False
    
    # Calculate bounds
    max_pos = sum(d for d in diffs if d > 0)
    max_neg = -sum(d for d in diffs if d < 0)
    offset = max_neg
    total_span = max_pos + max_neg + 1
    
    # If span is too large, fall back to small subset checks
    if total_span > 200000:
        # Check size 2
        for i in range(m):
            for j in range(i+1, m):
                if diffs[i] + diffs[j] == 0:
                    return True
        # Check size 3
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    if diffs[i] + diffs[j] + diffs[k] == 0:
                        return True
        return False
        
    target_bit = 1 << offset
    mask = (1 << total_span) - 1
    
    dp = [0] * (m + 1)
    dp[0] = target_bit
    
    for d in diffs:
        for k in range(m, 0, -1):
            if dp[k-1] == 0:
                continue
            if d >= 0:
                dp[k] |= (dp[k-1] << d) & mask
            else:
                dp[k] |= (dp[k-1] >> (-d)) & mask
            # Early exit
            if dp[k] & target_bit:
                return True
                
    return False

def _is_valid(C):
    for i in range(len(C)):
        if _check_violation(C, i):
            return False
    return True

def _fast_incremental_check(C, new_elem):
    """
    Fast check for whether adding new_elem to C creates any violation.
    Checks all subset sizes up to min(len(C)+1, 5) efficiently.
    For larger subsets, rely on the full check.
    """
    a = new_elem
    n = len(C)
    C_set = set(C)
    C_sorted = sorted(C)
    
    max_check_size = min(n + 1, 6)  # Check up to size 6 subsets
    
    # Check if new_elem is mean of any subset of existing elements
    for k in range(2, max_check_size + 1):
        if k > n:
            break
        # Need sum of k elements from C to equal k * a
        target_sum = k * a
        # Use recursive subset sum check for small k
        if _subset_sum_exists(C_sorted, k, target_sum):
            return False
    
    # Check if any C[i] is mean of subset containing new_elem
    for i in range(n):
        ci = C[i]
        others = [C[j] for j in range(n) if j != i]
        for k in range(2, min(len(others) + 1, max_check_size + 1)):
            # Need subset of size k from (others + [new_elem]) that sums to k * ci
            # Case 1: subset doesn't include new_elem -> already valid by assumption
            # Case 2: subset includes new_elem -> need k-1 elements from others summing to k*ci - new_elem
            if k == 1:
                continue
            target_sum = k * ci - new_elem
            if target_sum < 0 or target_sum > sum(others):
                continue
            if _subset_sum_exists(others, k - 1, target_sum):
                return False
    
    return True

def _subset_sum_exists(arr, k, target):
    """Check if there exists a subset of exactly k elements from arr that sums to target."""
    if k == 0:
        return target == 0
    if k > len(arr):
        return False
    if target < 0:
        return False
    # Memoization for small cases
    n = len(arr)
    # If target is too large or too small, quick reject
    if k > 0:
        min_possible = sum(arr[:k]) if n >= k else sum(arr)
        max_possible = sum(arr[-k:]) if n >= k else sum(arr)
        if target < min_possible or target > max_possible:
            return False
    
    # Use DP for small k
    if k <= 4:
        # Recursive with memo
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def dp(idx, remaining_k, remaining_sum):
            if remaining_k == 0:
                return remaining_sum == 0
            if idx >= n or remaining_k > n - idx:
                return False
            # Option 1: don't take arr[idx]
            if dp(idx + 1, remaining_k, remaining_sum):
                return True
            # Option 2: take arr[idx]
            if remaining_sum >= arr[idx] and dp(idx + 1, remaining_k - 1, remaining_sum - arr[idx]):
                return True
            return False
        return dp(0, k, target)
    else:
        # For larger k, use bitset DP
        # This is slower but more general
        offset = 0
        # We need sum of k elements = target
        # Use DP over positions and count
        # dp[count] = set of achievable sums
        # But sums can be large, so use set-based approach for moderate sizes
        if sum(arr) > 100000:
            # Fall back to checking only small subsets
            return False
        
        dp = [set() for _ in range(k + 1)]
        dp[0].add(0)
        for x in arr:
            for c in range(min(k, n), 0, -1):
                if not dp[c-1]:
                    continue
                new_sums = set()
                for s in dp[c-1]:
                    new_s = s + x
                    if new_s <= target:
                        new_sums.add(new_s)
                dp[c] |= new_sums
                if target in dp[c]:
                    return True
        return target in dp[k]

def _greedy_construct(order, time_limit):
    selected = []
    start = time.time()
    
    for x in order:
        if time.time() - start > time_limit:
            break
        if len(selected) >= 48:
            break
        
        if _fast_incremental_check(selected, x):
            C = selected + [x]
            # Full check
            if _is_valid(C):
                selected.append(x)
    
    return selected

def _local_search_improve(C, time_limit):
    """
    Try to improve set C by removing some elements and adding others.
    Returns improved set if found.
    """
    start = time.time()
    C = list(C)
    C_set = set(C)
    all_vals = set(range(1, 513))
    candidates = list(all_vals - C_set)
    
    improved = False
    while time.time() - start < time_limit and len(C) < 48:
        found = False
        # Try removing 1 and adding 2
        for i in range(len(C)):
            if time.time() - start > time_limit:
                break
            C_temp = C[:i] + C[i+1:]
            C_temp_set = set(C_temp)
            # Try adding 2 candidates
            for j in range(len(candidates)):
                if time.time() - start > time_limit:
                    break
                x1 = candidates[j]
                if x1 in C_temp_set:
                    continue
                C_temp1 = C_temp + [x1]
                if _fast_incremental_check(C_temp, x1):
                    for k in range(j+1, len(candidates)):
                        if time.time() - start > time_limit:
                            break
                        x2 = candidates[k]
                        if x2 in C_temp_set or x2 == x1:
                            continue
                        C_temp2 = C_temp1 + [x2]
                        if _fast_incremental_check(C_temp1, x2):
                            if _is_valid(C_temp2):
                                C = C_temp2
                                C_set = set(C)
                                candidates = list(all_vals - C_set)
                                found = True
                                improved = True
                                break
                    if found:
                        break
            if found:
                break
        
        if not found:
            # Try just adding 1
            for x in candidates:
                if time.time() - start > time_limit:
                    break
                if _fast_incremental_check(C, x):
                    C_new = C + [x]
                    if _is_valid(C_new):
                        C = C_new
                        C_set = set(C)
                        candidates = list(all_vals - C_set)
                        improved = True
                        break
    
    return C

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Pre-defined good orders
    orders = []
    
    # 1. Lo-Hi alternating
    lo, hi = 1, 512
    oh = []
    while lo <= hi:
        oh.append(lo)
        if lo != hi: oh.append(hi)
        lo += 1
        hi -= 1
    orders.append(oh)
    
    # 2. Hi-Lo alternating
    lo, hi = 1, 512
    oh = []
    while lo <= hi:
        oh.append(hi)
        if lo != hi: oh.append(lo)
        lo += 1
        hi -= 1
    orders.append(oh)
    
    # 3. Random orders
    for seed in range(30):
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        orders.append(order)
        
    # 4. Spread order
    base = list(range(1, 513, 8))
    random.Random(123).shuffle(base)
    remaining = [x for x in range(1, 513) if x not in set(base)]
    random.Random(456).shuffle(remaining)
    orders.append(base + remaining)
    
    # 5. Powers of 2 based
    powers = [2**i for i in range(10) if 2**i <= 512]
    random.Random(789).shuffle(powers)
    rest = [x for x in range(1, 513) if x not in set(powers)]
    random.Random(999).shuffle(rest)
    orders.append(powers + rest)
    
    # Run greedy
    for i, order in enumerate(orders):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 15:
            break
            
        time_limit = min(1.5, _DEADLINE - elapsed - 10)
        if time_limit < 0.5:
            break
        result = _greedy_construct(order, time_limit)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
            
        if best_size == 48:
            break
            
    # Local search improvement
    if best_size < 48 and time.time() - start_time < _DEADLINE - 20:
        time_limit = min(15.0, _DEADLINE - time.time() - 10)
        if time_limit > 2:
            improved = _local_search_improve(best_set, time_limit)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    # If we have more time, try more random orders
    if best_size < 48 and time.time() - start_time < _DEADLINE - 10:
        for seed in range(100, 200):
            if time.time() - start_time > _DEADLINE - 5:
                break
            rng = random.Random(seed)
            order = list(range(1, 513))
            rng.shuffle(order)
            result = _greedy_construct(order, 0.8)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result[:]
                if best_size == 48:
                    break
    
    if not best_set:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best_set
