import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 23))
    
    vals = [0] * 23
    for x in range(1, 23):
        vals[x] = L // x
        
    best_set = []
    
    def dfs(start_idx, current_diffs, current_set):
        nonlocal best_set
        
        if time.time() > deadline:
            return
            
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        remaining = 22 - start_idx + 1
        if len(current_set) + remaining <= len(best_set):
            return
            
        for x in range(start_idx, 23):
            v = vals[x]
            
            if v in current_diffs:
                continue
                
            current_set.append(x)
            
            new_diffs = set(current_diffs)
            # Optimization: list comprehension for speed?
            # new_diffs.update(d + v for d in current_diffs)
            # new_diffs.update(d - v for d in current_diffs)
            
            for d in current_diffs:
                new_diffs.add(d + v)
                new_diffs.add(d - v)
            
            dfs(x + 1, new_diffs, current_set)
            
            current_set.pop()
            
    dfs(1, {0}, [])
    
    # Fallback to seed if empty (should not happen)
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
