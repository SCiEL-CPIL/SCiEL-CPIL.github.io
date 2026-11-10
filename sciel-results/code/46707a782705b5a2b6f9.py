import math
from collections import defaultdict

def construct():
    LIMIT_M = 10**18
    
    # Use a smaller set of primes to allow deeper exploration
    primes = [2, 3, 5, 7, 11, 13]
    
    solutions = defaultdict(list)
    
    def dfs(idx, curr_M, curr_k):
        if curr_M > LIMIT_M:
            return
        
        if idx == len(primes):
            if curr_M > 1:
                if curr_k not in solutions[curr_M]:
                    solutions[curr_M].append(curr_k)
            return
        
        p = primes[idx]
        a = 0
        p_pow = 1
        sigma_val = 1
        while True:
            if a == 0:
                T = 1
                k_part = 1
            else:
                p_pow *= p
                sigma_val += p_pow
                T = p_pow * sigma_val
            
            new_M = curr_M * T
            if new_M > LIMIT_M:
                break
            
            new_k = curr_k * k_part
            dfs(idx + 1, new_M, new_k)
            
            a += 1
            if a > 60:  # Safety bound
                break
    
    dfs(0, 1, 1)
    
    best_M = None
    best_count = 0
    for M, ks in solutions.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
    
    if best_M is None:
        return {'M': 1, 'k': [1]}
    
    best_ks = list(set(solutions[best_M]))
    
    return {'M': best_M, 'k': best_ks}
