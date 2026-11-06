"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy: Precompute SPF up to 10000 for fast factorization. Use a segmented
    approach to compute largest prime factors for all numbers in range, then scan
    for consecutive pairs with minimal max LPF.
    """
    start_time = time.time()
    deadline = start_time + 85
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Precompute SPF up to 10000 (covers all prime factors <= 10000)
    SPF_LIMIT = 10000
    spf = np.arange(SPF_LIMIT + 1, dtype=np.int64)
    for i in range(2, int(math.isqrt(SPF_LIMIT)) + 1):
        if spf[i] == i:
            spf[i*i::i] = np.where(spf[i*i::i] == spf[i*i::i], i, spf[i*i::i])
    
    # Primes up to SPF_LIMIT for trial division of cofactors
    primes = [i for i in range(2, SPF_LIMIT + 1) if spf[i] == i]
    primes_np = np.array(primes, dtype=np.int64)
    
    def largest_prime_factor(n):
        """Compute largest prime factor of n using SPF + trial division for cofactor."""
        if n <= 1:
            return 1
        max_pf = 1
        # Factor out small primes using SPF
        while n > 1:
            if n <= SPF_LIMIT:
                p = spf[n]
            else:
                # Find smallest prime factor by trial division with primes
                p = n  # assume prime
                for q in primes_np:
                    if q * q > n:
                        break
                    if n % q == 0:
                        p = q
                        break
                # If no factor found, n is prime
                if p == n:
                    return max(max_pf, n)
            max_pf = max(max_pf, p)
            while n % p == 0:
                n //= p
        return max_pf
    
    # Phase 1: Quick scan to find a good initial bound
    # Sample every 1000th number to get a rough estimate
    best_n = N_MIN
    best_lpf = float('inf')
    
    # Dense scan with SPF-accelerated factorization
    # Process in chunks to allow deadline checks
    CHUNK = 50000
    for chunk_start in range(N_MIN, N_MAX + 1, CHUNK):
        if time.time() > deadline:
            break
        chunk_end = min(chunk_start + CHUNK, N_MAX + 1)
        
        # Compute LPF for each n in [chunk_start, chunk_end)
        for n in range(chunk_start, chunk_end):
            lpf_n = largest_prime_factor(n)
            if lpf_n >= best_lpf:
                continue
            lpf_np1 = largest_prime_factor(n + 1)
            max_lpf = max(lpf_n, lpf_np1)
            if max_lpf < best_lpf:
                best_lpf = max_lpf
                best_n = n
    
    # Phase 2: Local refinement around the best candidate
    # Search in a window around best_n for potentially better candidates
    if time.time() < deadline - 10:
        window = 5000
        search_lo = max(N_MIN, best_n - window)
        search_hi = min(N_MAX, best_n + window)
        
        for n in range(search_lo, search_hi + 1):
            if time.time() > deadline:
                break
            lpf_n = largest_prime_factor(n)
            if lpf_n >= best_lpf:
                continue
            lpf_np1 = largest_prime_factor(n + 1)
            max_lpf = max(lpf_n, lpf_np1)
            if max_lpf < best_lpf:
                best_lpf = max_lpf
                best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
