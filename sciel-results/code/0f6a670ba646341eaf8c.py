"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] minimizing the largest prime factor of n*(n+1).
    
    Strategy:
    1. Generate small primes up to a threshold P.
    2. Use a segmented sieve to mark P-smooth numbers in [10000, 1000000] by
       dividing out all prime factors <= P; a number is P-smooth iff remainder is 1.
    3. Find consecutive pairs (n, n+1) where both are P-smooth.
    4. Among these sparse candidates, compute the exact largest prime factor of n*(n+1)
       via trial division and return the n with the minimum such value.
    """
    LO = 10000
    HI = 1000000
    
    # Step 1: Generate small primes up to threshold P
    P = 200
    
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    small_primes = sieve_primes(P)
    
    # Step 2: Segmented sieve to find P-smooth numbers in [LO, HI]
    # For each segment, create an array of remainders after dividing out small primes.
    # A number is P-smooth iff its remainder after dividing by all primes <= P is 1.
    
    segment_size = 50000
    smooth_set = set()  # Store n values that are P-smooth
    
    for seg_start in range(LO, HI + 1, segment_size):
        seg_end = min(seg_start + segment_size - 1, HI)
        seg_len = seg_end - seg_start + 1
        
        # rem[i] corresponds to number seg_start + i
        rem = list(range(seg_start, seg_end + 1))
        
        for p in small_primes:
            # Find first multiple of p in [seg_start, seg_end]
            first = ((seg_start + p - 1) // p) * p
            for m in range(first, seg_end + 1, p):
                idx = m - seg_start
                while rem[idx] % p == 0:
                    rem[idx] //= p
        
        # Mark numbers that are P-smooth (remainder == 1)
        for i in range(seg_len):
            if rem[i] == 1:
                smooth_set.add(seg_start + i)
    
    # Step 3: Find consecutive pairs where both n and n+1 are P-smooth
    candidates = []
    
    for n in sorted(smooth_set):
        if n > LO and n <= HI and (n + 1) in smooth_set:
            # Both n and n+1 are P-smooth; compute exact largest prime factor
            def largest_prime_factor(x):
                if x <= 1:
                    return 1
                lpf = 1
                d = 2
                while d * d <= x:
                    if x % d == 0:
                        lpf = d
                        while x % d == 0:
                            x //= d
                    d += 1 if d == 2 else 2
                if x > 1:
                    lpf = x
                return lpf
            
            lpf_n = largest_prime_factor(n)
            lpf_n1 = largest_prime_factor(n + 1)
            lpf_product = max(lpf_n, lpf_n1)
            candidates.append((lpf_product, n))
    
    if not candidates:
        return 10000
    
    candidates.sort()
    return candidates[0][1]
# EVOLVE-BLOCK-END
