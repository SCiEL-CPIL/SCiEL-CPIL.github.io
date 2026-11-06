import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # Primes p == 1 mod 4
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 127, 137, 139, 149, 157, 173, 179, 181, 193, 197, 199, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 997]
    
    max_N = 10**18
    best_P = 0
    best_N = 1
    best_exponents = []
    
    # Recursive search for optimal exponents
    # We want to maximize P = prod(e_i + 1)
    # Subject to prod(p_i^e_i) <= max_N
    # e_1 >= e_2 >= ... >= e_k >= 0
    
    def search(idx, current_N, current_P, exponents):
        nonlocal best_P, best_N, best_exponents
        
        if time.time() > deadline - 10:
            return
            
        if current_P > best_P:
            best_P = current_P
            best_N = current_N
            best_exponents = exponents[:]
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Max exponent for this prime
        # current_N * p^e <= max_N
        # e <= log(max_N / current_N) / log(p)
        
        max_e = 0
        temp_N = current_N
        while temp_N * p <= max_N:
            temp_N *= p
            max_e += 1
            
        # We also need to consider that future exponents must be <= current exponent
        # To maximize P, we usually want e_i to be non-increasing.
        # However, since p_i are increasing, we can have e_i decrease.
        # If we set e_idx = e, then e_{idx+1} <= e.
        
        # Iterate possible exponents from max_e down to 0
        # But we can prune: if current_P * (max_e + 1) * (max possible future P) <= best_P, prune?
        # Simple pruning: if current_N * p^max_e * (product of next primes) > max_N?
        
        for e in range(max_e, -1, -1):
            new_N = current_N * (p ** e)
            if new_N > max_N:
                continue
                
            # Heuristic: if e is large, we can't have large exponents for subsequent primes.
            # But the loop handles that.
            
            search(idx + 1, new_N, current_P * (e + 1), exponents + [e])
            
            # Optimization: if e == 0, then all subsequent exponents must be 0.
            # So we can just check the base case.
            if e == 0:
                # No more primes can be used
                if current_P > best_P:
                    best_P = current_P
                    best_N = current_N
                    best_exponents = exponents[:]
                break

    search(0, 1, 1, [])
    
    if best_N < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Find pairs for best_N
    N = best_N
    pairs = []
    x_max = isqrt(N // 2)
    
    # Use NumPy for chunked search
    import numpy as np
    
    chunk_size = 10**6
    found = 0
    expected_pairs = best_P // 2 # Approximate
    
    for start in range(1, x_max + 1, chunk_size):
        end = min(start + chunk_size, x_max + 1)
        x_vals = np.arange(start, end, dtype=np.int64)
        x_sqs = x_vals * x_vals
        y_sqs = N - x_sqs
        
        # Filter y_sqs >= 0
        valid = y_sqs >= 0
        if not np.any(valid):
            continue
            
        y_sqs = y_sqs[valid]
        x_sqs = x_sqs[valid]
        
        y_vals = np.sqrt(y_sqs).astype(np.int64)
        
        # Check for perfect squares
        # Due to floating point precision, we might need to check y^2 and (y+1)^2
        # But N is int64, y is int64.
        # y^2 might overflow int64? No, y <= 10^9, y^2 <= 10^18.
        
        diffs = y_sqs - y_vals * y_vals
        is_sq = diffs == 0
        
        # Also check (y+1)^2 just in case
        diffs2 = y_sqs - (y_vals + 1) * (y_vals + 1)
        is_sq2 = diffs2 == 0
        
        # Combine
        is_sq_final = is_sq | is_sq2
        
        if np.any(is_sq_final):
            indices = np.where(is_sq_final)[0]
            for idx in indices:
                a = int(x_sqs[idx])
                b = int(y_sqs[idx])
                # a = x^2, b = y^2
                # We need a <= b. Since x <= x_max = sqrt(N/2), x^2 <= N/2, so a <= b.
                pairs.append([a, b])
                
    if not pairs:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': N, 'pairs': pairs}

# EVOLVE-BLOCK-END
