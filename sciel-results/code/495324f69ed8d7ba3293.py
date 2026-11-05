import time
import random
import sys

_DEADLINE = 85.0

def _check_violation_full(C, idx):
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

def _is_valid_full(C):
    for i in range(len(C)):
        if _check_violation_full(C, i):
            return False
    return True

def _fast_check(S, x):
    """
    Check if x can be added to S without creating violations of size 2, 3, or 4.
    S is a list. x is an int.
    Returns True if safe.
    """
    S_set = set(S)
    n = len(S)
    
    # 1. x is mean of 2 in S
    target = 2 * x
    for a in S:
        b = target - a
        if b in S_set and b != a:
            return False
        # if b == a, then 2a = 2x => a=x, but x not in S.
        
    # 2. x is mean of 3 in S
    target = 3 * x
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            c = target - a - b
            if c in S_set and c != a and c != b:
                return False
                
    # 3. x is mean of 4 in S
    # 4x = a+b+c+d
    # Use pair sums
    if n >= 4:
        target = 4 * x
        pair_sums = {}
        for i in range(n):
            for j in range(i+1, n):
                s = S[i] + S[j]
                if s not in pair_sums:
                    pair_sums[s] = (i, j)
        
        for i in range(n):
            for j in range(i+1, n):
                s = S[i] + S[j]
                need = target - s
                if need in pair_sums:
                    k, l = pair_sums[need]
                    # Ensure all 4 are distinct
                    if i != k and i != l and j != k and j != l:
                        return False

    # 4. y in S is mean of {x, a, b}
    # 3y = x + a + b
    for y in S:
        target = 3 * y - x
        # Need a, b in S, distinct, and distinct from y
        for i in range(n):
            a = S[i]
            if a == y: continue
            b = target - a
            if b in S_set and b != a and b != y:
                return False

    # 5. y in S is mean of {x, a, b, c}
    # 4y = x + a + b + c
    # 4y - x = a + b + c
    if n >= 3:
        for y in S:
            target = 4 * y - x
            # Need 3 distinct elements from S \ {y} summing to target
            # This is 3-sum on S \ {y}
            # O(N^2) per y.
            others = [v for v in S if v != y]
            m = len(others)
            if m < 3: continue
            others_set = set(others)
            for i in range(m):
                a = others[i]
                for j in range(i+1, m):
                    b = others[j]
                    c = target - a - b
                    if c in others_set and c != a and c != b:
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
        
        if _fast_check(selected, x):
            C = selected + [x]
            # Full check is expensive, so we only do it if we think we are close or at the end?
            # No, we must ensure validity.
            # But full check is O(N * DP).
            # Let's do full check only if we have time, or every few steps?
            # Actually, if _fast_check passes, it's very likely valid.
            # We will do a full check at the end of the greedy phase.
            selected.append(x)
            
    return selected

def _repair_set(C, time_limit):
    """
    If C is invalid, remove violators and try to fill.
    """
    start = time.time()
    C = C[:]
    C_set = set(C)
    
    # Find violators
    violators = []
    for i in range(len(C)):
        if _check_violation_full(C, i):
            violators.append(C[i])
            
    # Remove violators
    for v in violators:
        if v in C:
            C.remove(v)
            C_set.remove(v)
            
    # Try to fill up to 48
    all_numbers = set(range(1, 513))
    not_in_C = list(all_numbers - C_set)
    random.shuffle(not_in_C)
    
    for x in not_in_C:
        if time.time() - start > time_limit:
            break
        if len(C) >= 48:
            break
        if _fast_check(C, x):
            C.append(x)
            C_set.add(x)
            
    return C

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Try multiple random orders
    for seed in range(50):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5:
            break
            
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        
        # Greedy
        time_limit = min(2.0, _DEADLINE - elapsed - 2)
        result = _greedy_construct(order, time_limit)
        
        # Full Validation
        if result and _is_valid_full(result):
            if len(result) > best_size:
                best_size = len(result)
                best_set = result[:]
                
            if best_size == 48:
                break
        else:
            # Repair
            if result:
                repaired = _repair_set(result, 1.0)
                if repaired and _is_valid_full(repaired):
                    if len(repaired) > best_size:
                        best_size = len(repaired)
                        best_set = repaired[:]
                    if best_size == 48:
                        break

    if not best_set:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best_set
