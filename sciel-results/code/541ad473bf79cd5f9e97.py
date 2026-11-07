import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    import numpy as np
    
    def try_sieve(K_max):
        sigma = np.zeros(K_max + 1, dtype=np.int64)
        for d in range(1, K_max + 1):
            sigma[d::d] += d
        
        k_vals = np.arange(1, K_max + 1, dtype=np.int64)
        M_vals = k_vals * sigma[1:]
        
        del sigma
        
        valid = M_vals <= 10**18
        M_valid = M_vals[valid]
        k_valid = k_vals[valid]
        
        del M_vals, k_vals
        
        if len(M_valid) == 0:
            return None, []
        
        order = np.argsort(M_valid)
        M_sorted = M_valid[order]
        k_sorted = k_valid[order]
        
        del M_valid, k_valid, order
        
        diff = np.diff(M_sorted)
        run_starts = np.concatenate(([0], np.where(diff != 0)[0] + 1))
        run_ends = np.concatenate((run_starts[1:], [len(M_sorted)]))
        run_lengths = run_ends - run_starts
        
        max_idx = np.argmax(run_lengths)
        max_M = int(M_sorted[run_starts[max_idx]])
        
        k_solutions = k_sorted[run_starts[max_idx]:run_ends[max_idx]]
        k_list = k_solutions.tolist()
        
        sqrt_M = math.isqrt(max_M)
        k_list = [k for k in k_list if k <= sqrt_M]
        
        return max_M, k_list
    
    best_M = None
    best_k_list = []
    
    # Primary: K_max = 150M
    if time.time() < deadline - 10:
        M1, k1 = try_sieve(150_000_000)
        if M1 is not None and len(k1) > len(best_k_list):
            best_M = M1
            best_k_list = k1
    
    # Secondary: try larger if time permits
    if best_M is not None and len(best_k_list) < 10 and time.time() < deadline - 20:
        M2, k2 = try_sieve(250_000_000)
        if M2 is not None and len(k2) > len(best_k_list):
            best_M = M2
            best_k_list = k2
    
    if best_M is None or not best_k_list:
        return {'M': 1, 'k': [1]}
    
    # Final verification
    def sigma(n):
        if n == 1:
            return 1
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                factor_sum = 1
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    factor_sum += power
                result *= factor_sum
            p += 1 if p == 2 else 2
        if temp > 1:
            result *= (1 + temp)
        return result
    
    verified_k = []
    for k in best_k_list:
        if k * sigma(k) == best_M and k <= math.isqrt(best_M):
            verified_k.append(k)
    
    if not verified_k:
        return {'M': 1, 'k': [1]}
    
    return {'M': int(best_M), 'k': verified_k}
