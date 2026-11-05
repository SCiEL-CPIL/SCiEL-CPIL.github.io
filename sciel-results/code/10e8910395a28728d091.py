import time
import random
import sys

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
    a = new_elem
    n = len(C)
    C_set = set(C)
    
    # 1. new_elem is mean of 2 others
    for i in range(n):
        for j in range(i+1, n):
            if C[i] + C[j] == 2 * a:
                return False
                
    # 2. new_elem is mean of 3 others
    for i in range(n):
        for j in range(i+1, n):
            rem = 3 * a - C[i] - C[j]
            if rem in C_set:
                if rem != C[i] and rem != C[j]:
                    return False
                    
    # 3. C[i] is mean of {new_elem, y}
    for i in range(n):
        y = 2 * C[i] - a
        if y in C_set and y != C[i]:
            return False
            
    # 4. C[i] is mean of {new_elem, y, z}
    for i in range(n):
        target_sum = 3 * C[i] - a
        for j in range(n):
            if C[j] == C[i]:
                continue
            z = target_sum - C[j]
            if z in C_set and z != C[i] and z != C[j]:
                return False
                
    # 5. Check for subset size 4: new_elem is mean of 4 others
    # Sum of 4 others = 4*a
    # We need 4 distinct elements from C summing to 4*a
    if n >= 4:
        target4 = 4 * a
        # Use meet-in-the-middle for 4 elements
        half = n // 2
        left = C[:half]
        right = C[half:]
        
        # All pairs from left
        left_sums = {}
        for i in range(len(left)):
            for j in range(i+1, len(left)):
                s = left[i] + left[j]
                left_sums[s] = (i, j)
        
        # Check pairs from right
        for i in range(len(right)):
            for j in range(i+1, len(right)):
                s = right[i] + right[j]
                need = target4 - s
                if need in left_sums:
                    li, lj = left_sums[need]
                    # All 4 elements are distinct by construction (left and right are disjoint, indices distinct)
                    return False
        
        # Also check all 4 from left or all 4 from right if they have >= 4 elements
        if len(left) >= 4:
            # Check 4 from left
            for i in range(len(left)):
                for j in range(i+1, len(left)):
                    for k in range(j+1, len(left)):
                        rem = target4 - left[i] - left[j] - left[k]
                        # Need rem in left, distinct from i,j,k
                        for l in range(k+1, len(left)):
                            if left[l] == rem:
                                return False
        if len(right) >= 4:
            for i in range(len(right)):
                for j in range(i+1, len(right)):
                    for k in range(j+1, len(right)):
                        rem = target4 - right[i] - right[j] - right[k]
                        for l in range(k+1, len(right)):
                            if right[l] == rem:
                                return False
        
    # 6. Check C[i] is mean of {new_elem} + 3 others
    # Sum of 4 elements = 4*C[i], one is new_elem
    if n >= 3:
        for i in range(n):
            target4 = 4 * C[i]
            rem_target = target4 - a  # sum of 3 others from C
            # Need 3 distinct elements from C (not C[i]) summing to rem_target
            others = [C[j] for j in range(n) if j != i]
            m = len(others)
            if m >= 3:
                # Check all triples
                for x in range(m):
                    for y in range(x+1, m):
                        for z in range(y+1, m):
                            if others[x] + others[y] + others[z] == rem_target:
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
    """Try to improve the set by swapping elements or adding new ones."""
    start = time.time()
    C = C[:]
    C_set = set(C)
    all_numbers = set(range(1, 513))
    not_in_C = list(all_numbers - C_set)
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        rng = random.Random(int(time.time() * 1000) % 100000)
        rng.shuffle(not_in_C)
        
        # Try adding new elements
        for x in not_in_C:
            if time.time() - start > time_limit:
                break
            if len(C) >= 48:
                break
            if x in C_set:
                continue
            if _fast_incremental_check(C, x):
                C_new = C + [x]
                if _is_valid(C_new):
                    C = C_new
                    C_set.add(x)
                    not_in_C.remove(x)
                    improved = True
                    if len(C) >= 48:
                        break
        
        # Try swapping: remove one, add two
        if len(C) < 48 and time.time() - start < time_limit:
            for i in range(len(C)):
                if time.time() - start > time_limit:
                    break
                C_without_i = C[:i] + C[i+1:]
                C_set_without_i = C_set - {C[i]}
                not_in_C_temp = [x for x in not_in_C if x != C[i]]
                
                # Try to add 2 elements
                added = 0
                new_C = C_without_i[:]
                new_set = C_set_without_i.copy()
                for x in not_in_C_temp:
                    if time.time() - start > time_limit:
                        break
                    if added >= 2:
                        break
                    if x in new_set:
                        continue
                    if _fast_incremental_check(new_C, x):
                        C_temp = new_C + [x]
                        if _is_valid(C_temp):
                            new_C = C_temp
                            new_set.add(x)
                            added += 1
                
                if added >= 1:
                    if len(new_C) > len(C):
                        C = new_C
                        C_set = new_set.copy()
                        not_in_C = [x for x in range(1, 513) if x not in C_set]
                        improved = True
                    elif len(new_C) == len(C) and rng.random() < 0.3:
                        # Sometimes accept same size for diversity
                        C = new_C
                        C_set = new_set.copy()
                        not_in_C = [x for x in range(1, 513) if x not in C_set]
                        improved = True
    
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
    for seed in range(20):
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
    
    # Run greedy
    for i, order in enumerate(orders):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5:
            break
            
        time_limit = min(2.0, _DEADLINE - elapsed - 2)
        result = _greedy_construct(order, time_limit)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
            
        if best_size == 48:
            break
            
    # If we have time, try to improve with local search
    if best_size < 48 and time.time() - start_time < _DEADLINE - 15:
        # Try local search on the best set
        remaining_time = _DEADLINE - time.time() - 10
        if remaining_time > 2:
            improved = _local_search_improve(best_set, remaining_time)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved[:]
                
    # If we have time, try more random orders
    if best_size < 48 and time.time() - start_time < _DEADLINE - 10:
        for seed in range(100, 150):
            if time.time() - start_time > _DEADLINE - 5:
                break
            rng = random.Random(seed)
            order = list(range(1, 513))
            rng.shuffle(order)
            result = _greedy_construct(order, 1.0)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result[:]
                if best_size == 48:
                    break
                    
    if not best_set:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best_set
