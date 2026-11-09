import time
from math import factorial
from functools import lru_cache

def construct():
    start_time = time.time()
    target = factorial(12)
    
    # Get all divisors of target greater than 12
    divisors = set()
    limit = int(target**0.5)
    for i in range(1, limit + 1):
        if target % i == 0:
            if i > 12:
                divisors.add(i)
            other = target // i
            if other > 12:
                divisors.add(other)
    
    valid_divisors = sorted(divisors)
    div_set = set(valid_divisors)
    
    # Memoization: (rem, last_idx, count) -> bool
    # last_idx is index into valid_divisors, we pick factors with index > last_idx
    memo = {}
    
    def dfs(rem, last_idx, count):
        if time.time() - start_time > 55:
            return None
        
        if rem == 1:
            return []
        
        if count >= 12:
            return None
        
        # Check memo
        key = (rem, last_idx, count)
        if key in memo:
            return memo[key]
        
        # Find start index: first divisor > valid_divisors[last_idx] if last_idx >= 0
        start = last_idx + 1 if last_idx >= 0 else 0
        
        for i in range(start, len(valid_divisors)):
            d = valid_divisors[i]
            if d > rem:
                break
            if rem % d != 0:
                continue
            
            res = dfs(rem // d, i, count + 1)
            if res is not None:
                result = [d] + res
                memo[key] = result
                return result
        
        memo[key] = None
        return None
    
    solution = dfs(target, -1, 0)
    if solution:
        return solution
    
    return [target]
