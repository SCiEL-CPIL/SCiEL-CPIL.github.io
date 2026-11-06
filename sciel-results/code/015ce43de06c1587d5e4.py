"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy: Use a segmented sieve to identify intervals where consecutive numbers
    are composed only of small primes (smooth numbers). Then verify candidates using
    exact trial division on the smooth numbers found.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Precompute smallest prime factor (SPF) up to sqrt(N_MAX) for trial division
    limit = int(math.isqrt(N_MAX)) + 1
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Helper: get largest prime factor using SPF
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        max_pf = 1
        while n > 1:
            p = spf[n] if n <= limit else None
            if p is None:
                # n is prime or has a prime factor > limit
                # Check if n itself is prime
                is_prime = True
                for i in range(2, int(math.isqrt(n)) + 1):
                    if n % i == 0:
                        is_prime = False
                        break
                if is_prime:
                    return n
                else:
                    # Find smallest factor
                    for i in range(2, int(math.isqrt(n)) + 1):
                        if n % i == 0:
                            p = i
                            break
                    else:
                        return n
            max_pf = max(max_pf, p)
            while n % p == 0:
                n //= p
        return max_pf
    
    # Sieve for primes up to 1000 for smoothness check
    SMOOTH_LIMIT = 1000
    is_prime = [True] * (SMOOTH_LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(SMOOTH_LIMIT)) + 1):
        if is_prime[i]:
            for j in range(i * i, SMOOTH_LIMIT + 1, i):
                is_prime[j] = False
    small_primes = [i for i in range(2, SMOOTH_LIMIT + 1) if is_prime[i]]
    
    # Helper: check if number is P-smooth (all prime factors <= P)
    def is_p_smooth(n, P):
        if n <= 1:
            return True
        for p in small_primes:
            if p > P:
                break
            while n % p == 0:
                n //= p
            if n == 1:
                return True
        return n == 1
    
    # Segmented sieve to find smooth consecutive pairs
    SEGMENT_SIZE = 10000
    best_n = None
    best_lpf = float('inf')
    
    # First pass: scan for candidates where both n and n+1 are P-smooth
    for P in [50, 100, 200, 500, 1000]:
        if time.time() > deadline:
            break
        
        # For each segment, mark numbers that are P-smooth
        for seg_start in range(N_MIN, N_MAX + 1, SEGMENT_SIZE):
            if time.time() > deadline:
                break
            
            seg_end = min(seg_start + SEGMENT_SIZE, N_MAX + 1)
            smooth_flags = [False] * (seg_end - seg_start)
            
            # Mark smooth numbers in this segment
            for i, num in enumerate(range(seg_start, seg_end)):
                if is_p_smooth(num, P):
                    smooth_flags[i] = True
            
            # Find consecutive smooth pairs
            for i in range(len(smooth_flags) - 1):
                if smooth_flags[i] and smooth_flags[i + 1]:
                    n_candidate = seg_start + i
                    if N_MIN <= n_candidate <= N_MAX:
                        # Verify with exact trial division
                        lpf_n = largest_prime_factor(n_candidate)
                        lpf_np1 = largest_prime_factor(n_candidate + 1)
                        max_lpf = max(lpf_n, lpf_np1)
                        
                        if max_lpf < best_lpf:
                            best_lpf = max_lpf
                            best_n = n_candidate
    
    # Fallback: if no smooth pair found, do a direct scan of a subset
    if best_n is None:
        # Direct scan with early termination
        for n in range(N_MIN, N_MAX + 1):
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
    
    # Final safety: return best found or default
    if best_n is None:
        return N_MIN
    return best_n
# EVOLVE-BLOCK-END
