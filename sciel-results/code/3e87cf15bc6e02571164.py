"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    def compute_phi_segment(start, length, primes):
        """Compute phi for range [start, start+length-1] using given primes."""
        phi = np.arange(start, start + length, dtype=np.int64)
        for p in primes:
            if p * p > start + length - 1:
                break
            first = ((start + p - 1) // p) * p
            indices = np.arange(first - start, length, p)
            mask = phi[indices] % p == 0
            if np.any(mask):
                idx_masked = indices[mask]
                phi[idx_masked] -= phi[idx_masked] // p
        for i in range(length):
            if phi[i] > 1:
                phi[i] -= phi[i] // phi[i]
        return phi
    
    def sieve_primes(limit):
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.where(is_prime)[0]
    
    primes = sieve_primes(1000000)
    
    best_k = 1
    best_s = 1
    
    range_sizes = [100000, 200000, 500000, 1000000, 2000000]
    start_positions = [
        10**6, 10**7, 10**8, 10**9, 10**10, 10**11,
        5*10**6, 5*10**7, 5*10**8, 5*10**9, 5*10**10, 5*10**11,
        3*10**6, 7*10**6, 11*10**6, 13*10**6, 17*10**6,
        2*10**7, 3*10**7, 4*10**7, 6*10**7, 8*10**7, 9*10**7,
        2*10**8, 3*10**8, 4*10**8, 6*10**8, 7*10**8, 8*10**8, 9*10**8,
        2*10**9, 3*10**9, 4*10**9, 6*10**9, 7*10**9, 8*10**9, 9*10**9,
        2*10**10, 3*10**10, 4*10**10, 6*10**10, 7*10**10, 8*10**10, 9*10**10,
        2*10**11, 3*10**11, 4*10**11, 6*10**11, 7*10**11, 8*10**11, 9*10**11
    ]
    
    for size in range_sizes:
        if time.time() > deadline - 20:
            break
        for start in start_positions:
            if start + size > 10**12:
                continue
            if time.time() > deadline - 10:
                break
            
            phi_vals = compute_phi_segment(start, size, primes)
            
            # Sliding window with early termination
            seen = set()
            left = 0
            for right in range(size):
                val = phi_vals[right]
                while val in seen:
                    seen.discard(phi_vals[left])
                    left += 1
                seen.add(val)
                current_k = right - left + 1
                if current_k > best_k:
                    best_k = current_k
                    best_s = start + left
    
    # Verify the result
    if best_k > 1:
        verify_start = best_s
        verify_k = best_k
        seen = set()
        valid = True
        for i in range(verify_k):
            x = verify_start + i
            y = x
            result = x
            for p in primes:
                if p * p > y:
                    break
                if y % p == 0:
                    while y % p == 0:
                        y //= p
                    result -= result // p
            if y > 1:
                result -= result // y
            if result in seen:
                valid = False
                break
            seen.add(result)
        
        if not valid:
            return {'start': 1, 'length': 1}
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
