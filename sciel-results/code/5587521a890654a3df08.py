import math
import time
import numpy as np

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Increase search range to find more collisions
    # 20 million should be manageable in time and memory
    N = 20_000_000
    
    # Sieve for sigma(n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Compute M_k = k * sigma(k)
    ks = np.arange(1, N + 1, dtype=np.int64)
    M_vals = ks * sigma[1:N+1]
    
    # Filter M <= 10^18
    valid_mask = M_vals <= 10**18
    ks_valid = ks[valid_mask]
    M_valid = M_vals[valid_mask]
    
    # Find the M with the most k values
    # np.unique is generally faster than manual sorting for this purpose
    unique_M, counts, inverse = np.unique(M_valid, return_counts=True, return_inverse=True)
    
    # Fallback if no valid M found (should not happen)
    if len(counts) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    max_count = np.max(counts)
    
    # If we don't find at least 3, use fallback
    # We hope to find > 5 to beat the current score
    if max_count < 3:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Get the index of the best M
    best_M_idx = np.argmax(counts)
    best_M = int(unique_M[best_M_idx])
    
    # Extract k values for this M
    mask_best = (inverse == best_M_idx)
    k_candidates = ks_valid[mask_best]
    
    # Verify solutions with exact factorization
    def _sigma_verify(n):
        if n <= 0:
            return 0
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                sp = 1
                pk = 1
                while temp % p == 0:
                    temp //= p
                    pk *= p
                    sp += pk
                result *= sp
            p += 1
        if temp > 1:
            result *= (1 + temp)
        return result

    final_k = []
    for k in k_candidates:
        if k >= 1 and k <= math.isqrt(best_M) and k * _sigma_verify(k) == best_M:
            final_k.append(k)
            
    # Ensure we have at least the fallback number of solutions
    if len(final_k) < 3:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': list(final_k)}
