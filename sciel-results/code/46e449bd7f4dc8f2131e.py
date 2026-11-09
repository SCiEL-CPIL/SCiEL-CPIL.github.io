"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # We need to find n in [1000, 1000000] that minimizes the number of distinct
    # "small-prime factors" (primes strictly below 24) extracted from n+i for i=1..24.
    # The primes strictly below 24 are: 2, 3, 5, 7, 11, 13, 17, 19, 23.
    # For each number m, the "complete factor composed of primes strictly below 24" is the
    # largest divisor of m that has no prime factor >= 24. Let's call this S(m).
    # We want to minimize the size of the set {S(n+i) : i=1..24}.
    
    # Note: S(m) is unique for each m. If two numbers have the same S(m), they share the same
    # "small-prime part". However, S(m) can be 1 for many numbers (those coprime to 2*3*5*7*11*13*17*19*23).
    
    # Strategy:
    # 1. Precompute S(m) for m in [1001, 1000000 + 24].
    # 2. Use a sliding window of size 24 over the array S.
    # 3. For each window, count the number of distinct values in S[n+1..n+24].
    # 4. Track the minimum count and the corresponding n.
    
    N_MAX = 1000000
    N_MIN = 1000
    WINDOW_SIZE = 24
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute S(m) for m from 1001 to N_MAX + WINDOW_SIZE
    # S(m) = product of p^e where p is in small_primes and e is the exponent of p in m.
    
    # We can compute this efficiently.
    # Let M_start = N_MIN + 1
    # Let M_end = N_MAX + WINDOW_SIZE
    M_start = N_MIN + 1
    M_end = N_MAX + WINDOW_SIZE
    size = M_end - M_start + 1
    
    S = np.ones(size, dtype=np.int64)
    
    for p in small_primes:
        # For each multiple of p, multiply S by p, then p^2, etc.
        # It's easier to just iterate through multiples.
        # But doing it per prime and updating S is O(N log N) roughly.
        # Given N ~ 10^6, this is fast.
        
        # To be faster, we can compute the power of p dividing each multiple.
        # Or simply:
        # for j in range(0, size, p):
        #     val = M_start + j
        #     while val % p == 0:
        #         S[j] *= p
        #         val //= p
        # This is O(N * sum(1/p)) which is fine.
        
        # Even better: precompute the highest power of p dividing each number?
        # No, the above loop is simple and fast enough for 10^6.
        
        # Optimization: use NumPy vectorization where possible?
        # The while loop is hard to vectorize.
        # Let's stick to the Python loop but it might be slow for 9 primes * 10^6 / p.
        # Sum(1/p) for p in {2,3,5,7,11,13,17,19,23} is approx 1.
        # So total operations ~ 10^6. This is very fast.
        
        for j in range(0, size, p):
            val = M_start + j
            while val % p == 0:
                S[j] *= p
                val //= p
    
    # Now S[j] corresponds to m = M_start + j.
    # We want to slide a window of size 24 over S.
    # Window starting at index i corresponds to n such that n+1 = M_start + i => n = M_start + i - 1.
    # We require n >= N_MIN and n <= N_MAX.
    # So M_start + i - 1 >= N_MIN => i >= N_MIN - M_start + 1 = 1000 - 1001 + 1 = 0.
    # M_start + i - 1 <= N_MAX => i <= N_MAX - M_start + 1 = 1000000 - 1001 + 1 = 999000.
    # Also we need i + 24 - 1 < size => i + 23 < size => i < size - 23.
    # size = 1000000 + 24 - 1001 + 1 = 1000024.
    # size - 23 = 1000001.
    # So i ranges from 0 to 999000 (inclusive).
    
    min_distinct = float('inf')
    best_n = N_MIN
    
    # We need to count distinct values in S[i : i+24].
    # Doing this naively for each i is O(24) per window, total O(N * 24) ~ 24 * 10^6, which is fine.
    # However, using a set() in Python for 10^6 windows might be slightly slow but likely under 60s.
    # Let's try to optimize if needed.
    # A dictionary count is also O(24).
    
    # Let's just use a set for simplicity first.
    # To speed up, we can use a sliding window count.
    # But 24 is small.
    
    for i in range(0, N_MAX - N_MIN + 1):
        # Window is S[i : i+24]
        # n = M_start + i - 1
        # Check bounds: n is in [N_MIN, N_MAX]
        # i=0 -> n=1000. i=999000 -> n=1000000.
        
        # Count distinct
        # Using set is easy.
        distinct_count = len(set(S[i:i+24]))
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = N_MIN + i
            
            # Early exit if we find 1 distinct value?
            # If all S(n+i) are the same, that's the minimum possible (1).
            if min_distinct == 1:
                break
    
    return best_n

# EVOLVE-BLOCK-END
