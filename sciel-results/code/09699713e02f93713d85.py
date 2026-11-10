import math
from collections import defaultdict

def construct():
    # We search for M <= 10^18 with many k such that k*sigma(k) = M.
    # We generate k from a set of small primes.
    
    LIMIT_M = 10**18
    
    # Primes to include in the factorization of k
    # Using fewer primes allows for a faster search, and high-collision M values
    # are typically composed of small prime factors.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    solutions = defaultdict(list)
    
    # Precompute T(p, a) = p^a * sigma(p^a) and K(p, a) = p^a
    # T(p, 0) = 1, K(p, 0) = 1
    # T(p, a) = p^a * (p^(a+1)-1)/(p-1)
    
    max_a = 60 # Safe upper bound for any prime
    T_table = {}
    K_table = {}
    
    for p in primes:
        T_table[p] = [1]
        K_table[p] = [1]
        for a in range(1, max_a):
            p_pow_a = pow(p, a)
            # Check if p_pow_a * sigma(p^a) exceeds LIMIT_M? 
            # We will check in recursion, but precomputing helps.
            sigma_val = (pow(p, a + 1) - 1) // (p - 1)
            T_val = p_pow_a * sigma_val
            if T_val > LIMIT_M:
                break
            T_table[p].append(T_val)
            K_table[p].append(p_pow_a)
            
    def recurse(i, curr_M, curr_k):
        if curr_M > LIMIT_M:
            return
        
        if i == len(primes):
            if curr_M > 1:
                if curr_k not in solutions[curr_M]:
                    solutions[curr_M].append(curr_k)
            return
        
        p = primes[i]
        T_list = T_table[p]
        K_list = K_table[p]
        
        for a in range(len(T_list)):
            new_M = curr_M * T_list[a]
            if new_M > LIMIT_M:
                break
            new_k = curr_k * K_list[a]
            recurse(i + 1, new_M, new_k)
            
    recurse(0, 1, 1)
    
    best_M = None
    best_ks = []
    
    for M, ks in solutions.items():
        if len(ks) > len(best_ks):
            best_ks = ks
            best_M = M
            
    if best_M is None:
        return {'M': 1, 'k': [1]}
        
    # Ensure k_i <= sqrt(M)
    # For k > 1, sigma(k) >= k + 1 > k, so M = k*sigma(k) > k^2, so k < sqrt(M).
    # For k = 1, M = 1, sqrt(1) = 1.
    # So this condition is always satisfied for valid k.
    
    # Filter to be safe, though it should be redundant
    valid_ks = [k for k in best_ks if k * k <= best_M]
    
    return {'M': best_M, 'k': valid_ks}
