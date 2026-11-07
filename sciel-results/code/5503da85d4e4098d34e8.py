import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds safety margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    limit = 10**18
    
    # Precompute p^a and sigma(p^a) for each prime
    # We only need up to the point where p^a * sigma(p^a) > limit
    prime_data = []
    for p in primes:
        powers = []
        sigmas = []
        p_pow = 1
        sigma_val = 1
        # sigma(p^a) = 1 + p + ... + p^a
        # term = p^a * sigma(p^a)
        a = 0
        while True:
            # Check if we can add this prime power
            # We need to know the max a such that p^a * sigma(p^a) <= limit
            # But we also multiply with other terms, so we just generate all possible terms
            # and let the DFS prune based on current_M * term <= limit
            
            # Actually, we can just generate terms until term > limit
            # But term grows fast.
            
            # Let's just compute on the fly in DFS or precompute up to a reasonable limit
            # p^a * (p^(a+1)-1)/(p-1)
            
            # Let's precompute for a up to 60
            break
        # Better approach: compute in DFS
        prime_data.append(p)

    solutions = defaultdict(list)
    
    # To avoid recursion depth issues and overhead, we can use an iterative approach or optimized recursion
    # Recursion depth is len(primes) = 14, which is fine.
    
    def dfs(idx, current_M, current_k):
        nonlocal solutions
        if time.time() > deadline:
            return False # Signal timeout
        
        if idx == len(primes):
            solutions[current_M].append(current_k)
            return True
            
        p = primes[idx]
        
        # Try exponent 0
        if not dfs(idx + 1, current_M, current_k):
            return False
            
        p_pow = 1
        sigma_val = 1
        # term = p^a * sigma(p^a)
        # We iterate a >= 1
        for a in range(1, 100):
            p_pow *= p
            sigma_val = sigma_val * p + 1 # This is not correct for sigma update
            # sigma(p^a) = 1 + p + ... + p^a
            # It's easier to compute from scratch or use recurrence
            # sigma(p^a) = sigma(p^(a-1)) + p^a
            
            # Let's re-calculate sigma properly
            # Or just keep a running sum
            break
        
        # Let's refactor the inner loop
        p_pow = 1
        sigma_sum = 1
        for a in range(1, 100):
            p_pow *= p
            sigma_sum += p_pow
            term = p_pow * sigma_sum
            
            if current_M * term > limit:
                break
                
            if not dfs(idx + 1, current_M * term, current_k * p_pow):
                return False
                
        return True

    dfs(0, 1, 1)
    
    # Find the best M
    best_M = 1
    best_k_list = [1]
    
    for M, k_list in solutions.items():
        if len(k_list) > len(best_k_list):
            best_M = M
            best_k_list = k_list
            
    # Ensure k_i <= isqrt(M)
    # As established, k <= sqrt(M) is always true for k >= 1
    # But let's double check for k=1, M=1. 1 <= 1. OK.
    
    return {'M': best_M, 'k': best_k_list}
