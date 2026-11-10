import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    max_k = 10**9
    max_M = 10**18
    
    solutions = {}
    
    def dfs(idx, current_k, current_sigma):
        if time.time() > deadline:
            return
            
        if idx == len(primes):
            M = current_k * current_sigma
            if M <= max_M:
                if M not in solutions:
                    solutions[M] = []
                solutions[M].append(current_k)
            return
            
        p = primes[idx]
        
        # Try exponent 0
        dfs(idx + 1, current_k, current_sigma)
        
        # Try exponent e >= 1
        k_val = current_k
        sigma_factor = 1
        for e in range(1, 100):
            k_val *= p
            if k_val > max_k:
                break
                
            sigma_factor = sigma_factor * p + 1
            
            new_sigma = current_sigma * sigma_factor
            M_candidate = k_val * new_sigma
            
            if M_candidate > max_M:
                break
                
            dfs(idx + 1, k_val, new_sigma)

    dfs(0, 1, 1)
    
    best_M = 1
    best_k_list = [1]
    max_count = 1
    
    for M, k_list in solutions.items():
        if len(k_list) > max_count:
            max_count = len(k_list)
            best_M = M
            best_k_list = k_list
            
    return {'M': best_M, 'k': best_k_list}
