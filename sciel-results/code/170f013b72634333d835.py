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
    
    # For n=48, total_span is small enough for full DP always.
    # Fallback just in case, though unlikely to trigger.
    if total_span > 500000:
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
            # Early exit: if any subset size >= 2 sums to 0
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
            if _is_valid(C):
                selected.append(x)
    
    return selected

def _generate_orders():
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
    
    # 3. Powers of 2 first, then rest
    pows = [2**i for i in range(1, 10) if 2**i <= 512]
    rest = [x for x in range(1, 513) if x not in set(pows)]
    random.Random(42).shuffle(rest)
    orders.append(pows + rest)
    
    # 4. Spread orders
    for step in [5, 10, 20, 50]:
        base = list(range(1, 513, step))
        random.Random(step).shuffle(base)
        remaining = [x for x in range(1, 513) if x not in set(base)]
        random.Random(step+1).shuffle(remaining)
        orders.append(base + remaining)
        
    # 5. Random orders
    for seed in range(50):
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        orders.append(order)
        
    # 6. Cluster orders
    rng = random.Random(789)
    cluster_centers = list(range(10, 520, 30))
    rng.shuffle(cluster_centers)
    cluster_order = []
    for c in cluster_centers:
        spread = rng.randint(5, 15)
        for d in range(-spread, spread+1):
            x = c + d
            if 1 <= x <= 512:
                cluster_order.append(x)
    seen = set()
    unique_order = []
    for x in cluster_order:
        if x not in seen:
            seen.add(x)
            unique_order.append(x)
    remaining = [x for x in range(1, 513) if x not in seen]
    rng.shuffle(remaining)
    orders.append(unique_order + remaining)
    
    return orders

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    orders = _generate_orders()
    
    # Shuffle orders slightly to avoid bias towards first ones if they are bad?
    # No, structured ones are good. Keep them first.
    
    for i, order in enumerate(orders):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5:
            break
            
        time_limit = min(1.5, _DEADLINE - elapsed - 2)
        result = _greedy_construct(order, time_limit)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
            
        if best_size == 48:
            break
            
    # Try to improve best_set by adding any missing element
    if best_size < 48 and time.time() - start_time < _DEADLINE - 5:
        current_set = best_set[:]
        current_set_set = set(current_set)
        # Try adding any x from 1..512 not in set
        # Sort by "distance" from existing elements? Or just random?
        # Random is safer to avoid time spikes.
        candidates = [x for x in range(1, 513) if x not in current_set_set]
        random.Random(12345).shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > _DEADLINE - 2:
                break
            if _fast_incremental_check(current_set, x):
                new_set = current_set + [x]
                if _is_valid(new_set):
                    current_set = new_set
                    current_set_set.add(x)
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
                    if best_size == 48:
                        break
                        
    if not best_set:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best_set
