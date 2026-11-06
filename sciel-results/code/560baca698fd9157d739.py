import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    # Try to find minimal k
    # k=1, 2 impossible for d >= 5
    for k in range(3, 12):
        if time.time() > deadline:
            break
        sol = dfs_search(k, 5, Fraction(0, 1), k, 10000000, deadline)
        if sol:
            return sol
            
    # Fallback: Greedy construction
    # This might produce a larger k, but is safe
    return greedy_construct()

def dfs_search(k, min_d, sum_frac, remaining_terms, max_d, deadline):
    if remaining_terms == 0:
        if sum_frac == 1:
            return []
        return None
        
    if sum_frac >= 1:
        return None
        
    target = 1 - sum_frac
    if target <= 0:
        return None
        
    d_min = max(min_d, math.ceil(1 / target))
    
    # Upper bound: target < remaining_terms / d  => d < remaining_terms / target
    d_max = min(max_d, math.floor(remaining_terms / target))
    
    if d_min > d_max:
        return None
        
    for d in range(d_min, d_max + 1):
        new_sum = sum_frac + Fraction(1, d)
        if new_sum > 1:
            break
        res = dfs_search(k, d + 1, new_sum, remaining_terms - 1, max_d, deadline)
        if res is not None:
            return [d] + res
        if time.time() > deadline:
            return None
            
    return None

def greedy_construct():
    denominators = []
    remaining = Fraction(1, 1)
    last_d = 4
    while remaining > 0:
        d = max(last_d + 1, math.ceil(1 / remaining))
        if d > 10000000:
            break
        denominators.append(d)
        remaining -= Fraction(1, d)
        last_d = d
        if len(denominators) > 256:
            break
    return denominators
