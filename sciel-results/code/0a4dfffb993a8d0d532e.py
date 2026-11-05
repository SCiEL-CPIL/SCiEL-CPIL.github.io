import time
import random
import sys

_DEADLINE = 85.0
_MAX_SIZE = 48
_N = 512

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
    Fast check for whether adding new_elem to C creates a violation.
    Checks:
    1. new_elem is mean of subset of C (size 2, 3, 4)
    2. Any C[i] is mean of subset containing new_elem (size 2, 3)
    """
    a = new_elem
    n = len(C)
    C_set = set(C)
    
    # 1. new_elem is mean of 2 others: C[i] + C[j] = 2*a
    for i in range(n):
        for j in range(i+1, n):
            if C[i] + C[j] == 2 * a:
                return False
                
    # 2. new_elem is mean of 3 others: C[i] + C[j] + C[k] = 3*a
    for i in range(n):
        for j in range(i+1, n):
            rem = 3 * a - C[i] - C[j]
            if rem in C_set and rem != C[i] and rem != C[j]:
                return False
                    
    # 3. new_elem is mean of 4 others: C[i]+C[j]+C[k]+C[l] = 4*a
    # Use meet-in-the-middle: split into two pairs
    pair_sums = {}
    for i in range(n):
        for j in range(i+1, n):
            s = C[i] + C[j]
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((i, j))
    
    for i in range(n):
        for j in range(i+1, n):
            target = 4 * a - C[i] - C[j]
            if target in pair_sums:
                for (k, l) in pair_sums[target]:
                    # Ensure all 4 indices are distinct
                    if len({i, j, k, l}) == 4:
                        return False
                        
    # 4. C[i] is mean of {new_elem, y}: C[i] = (a + y) / 2 => y = 2*C[i] - a
    for i in range(n):
        y = 2 * C[i] - a
        if y in C_set and y != C[i]:
            return False
            
    # 5. C[i] is mean of {new_elem, y, z}: 3*C[i] = a + y + z => y + z = 3*C[i] - a
    for i in range(n):
        target_sum = 3 * C[i] - a
        for j in range(n):
            if C[j] == C[i]:
                continue
            z = target_sum - C[j]
            if z in C_set and z != C[i] and z != C[j]:
                return False
                
    return True

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
    Try to improve C by swapping elements or adding/removing.
    Returns improved set.
    """
    start = time.time()
    C = C[:]
    n = len(C)
    
    # Try to add elements
    for x in range(1, _N + 1):
        if time.time() - start > time_limit:
            break
        if x in C:
            continue
        if len(C) >= _MAX_SIZE:
            break
        if _fast_incremental_check(C, x):
            C_new = C + [x]
            if _is_valid(C_new):
                C = C_new
    
    # Try swaps: remove one, try to add another
    for i in range(len(C)):
        if time.time() - start > time_limit:
            break
        C_without_i = C[:i] + C[i+1:]
        for x in range(1, _N + 1):
            if time.time() - start > time_limit:
                break
            if x in C_without_i:
                continue
            if _fast_incremental_check(C_without_i, x):
                C_candidate = C_without_i + [x]
                if _is_valid(C_candidate):
                    # Check if this is an improvement (it's same size, so just accept)
                    C = C_candidate
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
    base = list(range(1, 513, 10))
    random.Random(123).shuffle(base)
    remaining = [x for x in range(1, 513) if x not in set(base)]
    random.Random(456).shuffle(remaining)
    orders.append(base + remaining)
    
    # 5. Power-of-2 first
    pow2 = [1 << i for i in range(10) if (1 << i) <= 512]
    rest = [x for x in range(1, 513) if x not in set(pow2)]
    random.Random(789).shuffle(rest)
    orders.append(pow2 + rest)
    
    # 6. Fibonacci-like
    fib = [1]
    while fib[-1] <= 512:
        fib.append(fib[-1] + fib[-2] if len(fib) >= 2 else 2)
    fib = list(set(fib))
    rest = [x for x in range(1, 513) if x not in set(fib)]
    random.Random(321).shuffle(rest)
    orders.append(fib + rest)
    
    # Run greedy
    for i, order in enumerate(orders):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 10:
            break
            
        time_limit = min(1.5, _DEADLINE - elapsed - 8)
        result = _greedy_construct(order, time_limit)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
            
        if best_size == _MAX_SIZE:
            break
            
    # If we have time, try local search to improve
    if best_size < _MAX_SIZE and time.time() - start_time < _DEADLINE - 5:
        remaining_time = _DEADLINE - time.time() - 3
        if remaining_time > 2:
            improved = _local_search_improve(best_set, remaining_time)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    # Try more random orders if time permits
    if best_size < _MAX_SIZE and time.time() - start_time < _DEADLINE - 5:
        for seed in range(100, 200):
            if time.time() - start_time > _DEADLINE - 3:
                break
            rng = random.Random(seed)
            order = list(range(1, 513))
            rng.shuffle(order)
            result = _greedy_construct(order, 0.8)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result[:]
                if best_size == _MAX_SIZE:
                    break
    
    if not best_set:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best_set
