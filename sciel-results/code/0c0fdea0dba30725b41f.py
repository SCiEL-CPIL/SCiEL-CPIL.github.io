import time
from math import gcd
from functools import reduce
import random

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

_lcm_cache = {}
def _lcm_list_cached(lst):
    key = tuple(sorted(lst))
    if key in _lcm_cache:
        return _lcm_cache[key]
    result = _lcm_list(lst)
    _lcm_cache[key] = result
    return result

def _subset_sum_exists(target, weights):
    """
    Check if target can be formed by a non-empty subset of weights.
    Uses meet-in-the-middle.
    """
    n = len(weights)
    if n == 0:
        return False
    if target <= 0:
        return False
    if target > sum(weights):
        return False
    
    # If target is one of the weights
    if target in weights:
        return True

    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate subset sums for w1
    sums1 = set([0])
    for w in w1:
        new_sums = set()
        for s in sums1:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        sums1.update(new_sums)
        
    # Generate subset sums for w2
    sums2 = set([0])
    for w in w2:
        new_sums = set()
        for s in sums2:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        sums2.update(new_sums)
        
    # Check if any pair sums to target
    for s1 in sums1:
        if (target - s1) in sums2:
            return True
    return False

def _can_add(k, selected):
    """
    Check if k can be added to selected set S.
    """
    if not selected:
        return True
        
    # Check 1: Is 1/k a subsum of S?
    L_S = _lcm_list_cached(selected)
    target_1 = L_S // k
    if target_1 == 0:
        # L_S < k? Impossible since k <= 24 and L_S is LCM of [1..24] subset
        pass
    weights_1 = [L_S // y for y in selected]
    if _subset_sum_exists(target_1, weights_1):
        return False
        
    # Check 2: For each x in S, does adding k create a subsum for x?
    # We need 1/x = sum(T) + 1/k => 1/x - 1/k = sum(T)
    # T is subset of S \ {x}
    for x in selected:
        if x >= k:
            # 1/x <= 1/k, so 1/x - 1/k <= 0. Sum of positive reciprocals > 0.
            continue
            
        S_prime = [y for y in selected if y != x]
        if not S_prime:
            continue
            
        L_S_prime = _lcm_list_cached(S_prime)
        
        # Target: 1/x - 1/k = (k - x) / (x * k)
        # We want L_S_prime * (1/x - 1/k) to be an integer sum of L_S_prime/y
        # L_S_prime * (k-x) / (x*k)
        numerator = L_S_prime * (k - x)
        denominator = x * k
        
        if numerator % denominator != 0:
            continue
            
        target_2 = numerator // denominator
        if target_2 <= 0:
            continue
            
        weights_2 = [L_S_prime // y for y in S_prime]
        
        if _subset_sum_exists(target_2, weights_2):
            return False
            
    return True

def construct():
    global _lcm_cache
    _lcm_cache = {}
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy Descending
    S = []
    for c in range(24, 0, -1):
        if _time_remaining() < 5.0:
            break
        if _can_add(c, S):
            S.append(c)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    # Strategy 2: Greedy Ascending
    S = []
    for c in range(1, 25):
        if _time_remaining() < 5.0:
            break
        if _can_add(c, S):
            S.append(c)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    # Strategy 3: Randomized Greedy
    random.seed(42)
    attempts = 0
    while _time_remaining() > 10.0 and attempts < 100:
        attempts += 1
        perm = list(range(1, 25))
        random.shuffle(perm)
        S = []
        for c in perm:
            if _time_remaining() < 5.0:
                break
            if _can_add(c, S):
                S.append(c)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            if best_size == 24:
                break
                
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Result: {res}, Size: {len(res)}")
