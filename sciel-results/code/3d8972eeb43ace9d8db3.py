"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, 5s safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    
    def largest_prime_factor(n):
        """Compute largest prime factor of n by trial division."""
        if n <= 1:
            return 1
        result = 1
        # Check factor 2
        while n % 2 == 0:
            result = 2
            n //= 2
        # Check odd factors
        f = 3
        while f * f <= n:
            while n % f == 0:
                result = f
                n //= f
            f += 2
        if n > 1:
            result = n
        return result
    
    # Strategy: Use segmented sieve to find smooth numbers, then verify candidates
    # First, let's do a quick scan with a moderate smoothness threshold
    
    # Precompute primes up to a reasonable limit for trial division
    # We need primes up to sqrt(1000000) = 1000 for trial division
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve_primes(1000)
    
    def largest_prime_factor_fast(n, primes):
        """Compute largest prime factor using precomputed primes."""
        if n <= 1:
            return 1
        result = 1
        for p in primes:
            if p * p > n:
                break
            while n % p == 0:
                result = p
                n //= p
        if n > 1:
            result = n
        return result
    
    # Approach: Scan for consecutive pairs where both n and n+1 are "smooth"
    # i.e., have small largest prime factors
    # We'll use a two-pass approach:
    # Pass 1: Quick scan with a threshold to find promising regions
    # Pass 2: Detailed verification in those regions
    
    best_lpf = float('inf')
    best_n = N_MIN
    
    # First, let's try a segmented approach
    # For each n in range, compute LPF of n and n+1, track the max
    # But doing this for all n is expensive. Instead, let's be smarter.
    
    # Key insight: if n and n+1 are both smooth (all prime factors <= P),
    # then the LPF of n(n+1) is at most P.
    # So we want to find consecutive pairs that are both P-smooth for small P.
    
    # Let's try increasing P until we find such pairs
    # For a given P, we can sieve to find P-smooth numbers in [N_MIN, N_MAX]
    
    def find_smooth_consecutive_pairs(P, N_min, N_max):
        """Find consecutive pairs in [N_min, N_max] that are both P-smooth."""
        # A number is P-smooth if all its prime factors are <= P
        # We can check this by dividing out all primes <= P
        # and seeing if the remainder is 1
        
        # For efficiency, we'll check each number
        # But we can optimize: if n is P-smooth, then n+1 must also be P-smooth
        # Let's find all P-smooth numbers in range, then check for consecutive pairs
        
        smooth_set = set()
        for n in range(N_min, N_max + 1):
            temp = n
            for p in primes:
                if p > P:
                    break
                if p * p > temp:
                    break
                while temp % p == 0:
                    temp //= p
            if temp == 1 or (temp <= P):
                smooth_set.add(n)
        
        # Find consecutive pairs
        pairs = []
        for n in sorted(smooth_set):
            if n + 1 in smooth_set:
                pairs.append(n)
        return pairs
    
    # Try different smoothness thresholds
    # Start with small P and increase
    found_any = False
    for P in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]:
        if time.time() > deadline - 5:
            break
        
        pairs = find_smooth_consecutive_pairs(P, N_MIN, N_MAX)
        if pairs:
            found_any = True
            # Verify these candidates with exact LPF computation
            for n in pairs:
                if time.time() > deadline - 3:
                    break
                lpf_n = largest_prime_factor_fast(n, primes)
                lpf_np1 = largest_prime_factor_fast(n + 1, primes)
                max_lpf = max(lpf_n, lpf_np1)
                if max_lpf < best_lpf:
                    best_lpf = max_lpf
                    best_n = n
            # If we found pairs with small P, we can potentially stop early
            # but let's continue to see if we can do better
            if best_lpf <= P:
                # We can't do better than P if we're using P-smooth numbers
                # But there might be non-smooth numbers with smaller max LPF
                # So continue searching
                pass
    
    # If we haven't found anything good, do a more thorough scan
    if not found_any or best_lpf > 100:
        # Scan a subset of the range more carefully
        # Focus on regions where numbers might be smooth
        for n in range(N_MIN, N_MAX + 1):
            if time.time() > deadline - 3:
                break
            lpf_n = largest_prime_factor_fast(n, primes)
            lpf_np1 = largest_prime_factor_fast(n + 1, primes)
            max_lpf = max(lpf_n, lpf_np1)
            if max_lpf < best_lpf:
                best_lpf = max_lpf
                best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
