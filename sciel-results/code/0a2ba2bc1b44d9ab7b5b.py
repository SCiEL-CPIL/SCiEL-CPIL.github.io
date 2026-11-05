import time
import random
import sys

_DEADLINE = 85.0
_MAX_SIZE = 48
_N = 512

def _check_violation(C, idx):
    """
    Check if C[idx] is the mean of any subset of >=2 OTHER elements.
    Uses bitset DP with sorted-bounds pruning.
    """
    a = C[idx]
    n = len(C)
    if n < 3:
        return False
    
    diffs = [C[j] - a for j in range(n) if j != idx]
    m = len(diffs)
    if m < 2:
        return False
    
    # Sort diffs for bounds pruning
    diffs_sorted = sorted(diffs)
    
    # Precompute prefix sums of sorted diffs for bounds
    prefix = [0] * (m + 1)
    for i in range(m):
        prefix[i+1] = prefix[i] + diffs_sorted[i]
    
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
        # Check size 4
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    for l in range(k+1, m):
                        if diffs[i] + diffs[j] + diffs[k] + diffs[l] == 0:
                            return True
        return False
        
    target_bit = 1 << offset
    mask = (1 << total_span) - 1
    
    dp = [0] * (m + 1)
    dp[0] = target_bit
    
    # Track which subset sizes are still possible using bounds
    # For subset size k, the min sum is sum of k smallest diffs
    # and max sum is sum of k largest diffs
    # If 0 is not in [min_sum, max_sum], skip that k
    
    for d in diffs:
        for k in range(m, 0, -1):
            if dp[k-1] == 0:
                continue
            # Bounds check: can we reach 0 with exactly k elements?
            # Min possible sum with k elements: sum of k smallest
            # Max possible sum with k elements: sum of k largest
            min_sum_k = prefix[k]  # k smallest
            max_sum_k = prefix[m] - prefix[m-k]  # k largest
            if 0 < min_sum_k or 0 > max_sum_k:
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
    Uses O(n) or O(n^2) checks with sets for efficiency.
    """
    a = new_elem
    n = len(C)
    C_set = set(C)
    
    # 1. new_elem is mean of 2 others: C[i] + C[j] = 2*a
    # For each C[i], check if 2*a - C[i] is in C_set and != C[i]
    for i in range(n):
        complement = 2 * a - C[i]
        if complement in C_set and complement != C[i]:
            return False
            
    # 2. new_elem is mean of 3 others: C[i] + C[j] + C[k] = 3*a
    # For each pair (i,j), check if 3*a - C[i] - C[j] is in C_set
    for i in range(n):
        for j in range(i+1, n):
            rem = 3 * a - C[i] - C[j]
            if rem in C_set and rem != C[i] and rem != C[j]:
                return False
                
    # 3. new_elem is mean of 4 others: C[i]+C[j]+C[k]+C[l] = 4*a
    # Use meet-in-the-middle with sets
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
                    if len({i, j, k, l}) == 4:
                        return False
                        
    # 4. C[i] is mean of {new_elem, y}: y = 2*C[i] - a
    for i in range(n):
        y = 2 * C[i] - a
        if y in C_set and y != C[i] and y != a:
            return False
            
    # 5. C[i] is mean of {new_elem, y, z}: y + z = 3*C[i] - a
    for i in range(n):
        target_sum = 3 * C[i] - a
        for j in range(n):
            if C[j] == C[i]:
                continue
            z = target_sum - C[j]
            if z in C_set and z != C[i] and z != C[j] and z != a:
                return False
                
    # 6. C[i] is mean of {new_elem, y, z, w}: y+z+w = 4*C[i]-a
    # This is expensive; do a bounded check
    for i in range(n):
        target_sum = 4 * C[i] - a
        ci = C[i]
        others = [C[j] for j in range(n) if C[j] != ci]
        mo = len(others)
        if mo < 3:
            continue
        # Use set-based check for triples
        others_set = set(others)
        for j in range(mo):
            for k in range(j+1, mo):
                w = target_sum - others[j] - others[k]
                if w in others_set and w != others[j] and w != others[k] and w != a:
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

def _count_violations(C):
    """
    Count the number of violated mean constraints in C.
    Returns (count, list_of_violated_indices)
    """
    violated = []
    for i in range(len(C)):
        if _check_violation(C, i):
            violated.append(i)
    return len(violated), violated

def _simulated_annealing_polish(C, time_limit):
    """
    Simulated annealing to minimize constraint violations.
    Tries swaps and additions/removals to improve the set.
    """
    start = time.time()
    C = C[:]
    C_set = set(C)
    
    # Initial state
    curr_viol, _ = _count_violations(C)
    best_C = C[:]
    best_viol = curr_viol
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() - start < time_limit:
        if T < T_min:
            break
            
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.3 and len(C) < _MAX_SIZE:
            # Add a random element
            candidates = [x for x in range(1, _N + 1) if x not in C_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            C_new = C + [x]
            C_new_set = C_set | {x}
            
        elif move_type < 0.7:
            # Swap: remove one, add another
            if not C:
                continue
            i = random.randint(0, len(C) - 1)
            candidates = [x for x in range(1, _N + 1) if x not in C_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            C_new = C[:i] + C[i+1:] + [x]
            C_new_set = C_set - {C[i]} | {x}
            
        elif move_type < 0.85:
            # Remove one element
            if len(C) <= 2:
                continue
            i = random.randint(0, len(C) - 1)
            C_new = C[:i] + C[i+1:]
            C_new_set = C_set - {C[i]}
            
        else:
            # Try to add a specific promising element
            candidates = [x for x in range(1, _N + 1) if x not in C_set]
            if not candidates or len(C) >= _MAX_SIZE:
                continue
            x = random.choice(candidates[:20])  # Prefer smaller elements
            C_new = C + [x]
            C_new_set = C_set | {x}
        
        # Evaluate new state
        new_viol, _ = _count_violations(C_new)
        delta = new_viol - curr_viol
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            # Accept move
            C = C_new
            C_set = C_new_set
            curr_viol = new_viol
            
            if curr_viol < best_viol:
                best_viol = curr_viol
                best_C = C[:]
        else:
            # Reject move
            pass
        
        # Cool down
        T *= cooling
    
    # Final greedy add pass
    if time.time() - start < time_limit - 2:
        for x in range(1, _N + 1):
            if time.time() - start > time_limit - 1:
                break
            if x in C_set:
                continue
            if len(C) >= _MAX_SIZE:
                break
            if _fast_incremental_check(C, x):
                C_new = C + [x]
                if _is_valid(C_new):
                    C = C_new
                    C_set = set(C)
                    if len(C) > len(best_C):
                        best_C = C[:]
    
    # Return the best found
    if len(best_C) > len(C):
        return best_C
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
    
    # 7. Mid-range first (200-300), then spread out
    mid = list(range(200, 301))
    random.Random(4567).shuffle(mid)
    rest = [x for x in range(1, 513) if x not in set(mid)]
    random.Random(8901).shuffle(rest)
    orders.append(mid + rest)
    
    # 8. Primes first
    def _is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i+2) == 0:
                return False
            i += 6
        return True
    primes = [x for x in range(1, 513) if _is_prime(x)]
    rest = [x for x in range(1, 513) if x not in set(primes)]
    random.Random(2468).shuffle(rest)
    orders.append(primes + rest)
    
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
    if best_size < _MAX_SIZE and time.time() - start_time < _DEADLINE - 10:
        remaining_time = _DEADLINE - time.time() - 5
        if remaining_time > 3:
            improved = _local_search_improve(best_set, min(remaining_time, 15))
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    # Try simulated annealing polish
    if best_size < _MAX_SIZE and time.time() - start_time < _DEADLINE - 5:
        remaining_time = _DEADLINE - time.time() - 3
        if remaining_time > 5:
            polished = _simulated_annealing_polish(best_set, remaining_time)
            if len(polished) > best_size:
                best_size = len(polished)
                best_set = polished
    
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
