"""Program for large reciprocal-subsum-free sets using integer Diophantine subset-sum."""
import time
from math import gcd
from functools import reduce

_START_TIME = time.time()
_DEADLINE = 85.0  # Tighter safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _is_subset_sum_integer(target_int, weights):
    """
    Check if target_int can be formed by a sum of a non-empty subset of weights.
    weights are positive integers.
    Uses meet-in-the-middle for efficiency.
    """
    n = len(weights)
    if n == 0:
        return False
    
    # Quick checks
    total = sum(weights)
    if target_int > total:
        return False
    if target_int <= 0:
        return False
        
    # If target is exactly one of the weights
    if target_int in weights:
        return True
    
    # Meet in the middle
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate all subset sums for w1
    sums1 = set([0])
    for w in w1:
        new_sums = set()
        for s in sums1:
            ns = s + w
            if ns <= target_int:
                new_sums.add(ns)
        sums1.update(new_sums)
    
    # Generate all subset sums for w2
    sums2 = set([0])
    for w in w2:
        new_sums = set()
        for s in sums2:
            ns = s + w
            if ns <= target_int:
                new_sums.add(ns)
        sums2.update(new_sums)
    
    # Check if any pair sums to target
    # target = s1 + s2 => s2 = target - s1
    for s1 in sums1:
        if (target_int - s1) in sums2:
            return True
    return False

_lcm_cache = {}

def _lcm_list_cached(lst):
    """Cached LCM computation."""
    key = tuple(sorted(lst))
    if key in _lcm_cache:
        return _lcm_cache[key]
    result = _lcm_list(lst)
    _lcm_cache[key] = result
    return result

def _can_add(k, selected):
    """
    Check if k can be added to selected set S.
    Condition:
    1. 1/k is not a subsum of S.
    2. For all x in S, 1/x is not a subsum of (S \ {x}) U {k}.
       This implies 1/x != sum(T) + 1/k for any T subset of S\{x}.
       => 1/x - 1/k != sum(T) for any T subset of S\{x}.
       => If 1/x > 1/k, check if (1/x - 1/k) is a subsum of S\{x}.
    """
    if not selected:
        return True
        
    # Scale factor for S
    L_S = _lcm_list_cached(selected)
    
    # Check 1: Is 1/k a subsum of S?
    # 1/k = sum(1/y for y in T) => L_S/k = sum(L_S/y for y in T)
    target_1 = L_S // k
    weights_1 = [L_S // y for y in selected]
    if _is_subset_sum_integer(target_1, weights_1):
        return False
        
    # Check 2: For each x in S
    for x in selected:
        if x >= k:
            # 1/x <= 1/k, so 1/x - 1/k <= 0. Sum of positive reciprocals > 0.
            # So 1/x - 1/k cannot be a subsum.
            continue
            
        # 1/x > 1/k
        # We need to check if 1/x - 1/k is a subsum of S \ {x}.
        S_prime = [y for y in selected if y != x]
        if not S_prime:
            continue
            
        L_S_prime = _lcm_list_cached(S_prime)
        
        # Target: 1/x - 1/k = (k - x) / (x * k)
        numerator = L_S_prime * (k - x)
        denominator = x * k
        
        if numerator % denominator != 0:
            continue
            
        target_2 = numerator // denominator
        weights_2 = [L_S_prime // y for y in S_prime]
        
        if _is_subset_sum_integer(target_2, weights_2):
            return False
            
    return True

def construct():
    """
    Construct the largest subset of [1, 24] that is reciprocal-subsum-free.
    """
    global _lcm_cache
    _lcm_cache = {}  # Clear cache at start
    
    S = []
    # Descending order first (large numbers are "safer" - less likely to be subsums)
    candidates = list(range(24, 0, -1))
    
    for c in candidates:
        if _time_remaining() < 3.0:
            break
        if _can_add(c, S):
            S.append(c)
            
    best_set = S
    best_size = len(S)
    
    # Try structured orderings first
    import random
    random.seed(42)
    
    structured_orders = [
        list(range(1, 25)),           # Ascending
        list(range(24, 0, -1)),       # Descending (already tried)
        [1, 24, 2, 23, 3, 22, 4, 21, 5, 20, 6, 19, 7, 18, 8, 17, 9, 16, 10, 15, 11, 14, 12, 13],  # Alternating
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # Descending again
    ]
    
    for perm in structured_orders:
        if _time_remaining() < 5.0:
            break
        S_curr = []
        for c in perm:
            if _time_remaining() < 3.0:
                break
            if _can_add(c, S_curr):
                S_curr.append(c)
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_set = S_curr
            if best_size == 24:
                break
    
    # Random restarts
    attempts = 0
    while _time_remaining() > 8.0 and attempts < 80:
        attempts += 1
        perm = list(range(1, 25))
        random.shuffle(perm)
        S_curr = []
        for c in perm:
            if _time_remaining() < 3.0:
                break
            if _can_add(c, S_curr):
                S_curr.append(c)
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_set = S_curr
            if best_size == 24:
                break
                
    return best_set

# Quick self-test
if __name__ == "__main__":
    res = construct()
    print(f"Result: {res}, Size: {len(res)}")
    
    # Verify
    def verify(S):
        S = set(S)
        for x in S:
            others = [y for y in S if y != x]
            # Check if 1/x is a subsum of others
            # Use integer check
            if not others:
                continue
            L = _lcm_list(others)
            target = L // x
            weights = [L // y for y in others]
            if _is_subset_sum_integer(target, weights):
                return False
        return True
        
    print(f"Valid: {verify(res)}")
