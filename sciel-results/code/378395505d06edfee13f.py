import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    # Precompute primes up to 10^7 for sieving and as candidates
    # 10^7 is a good balance: ~664k primes, sieve is fast
    P_MAX = 10_000_000
    is_prime_sieve = bytearray(b'\x01') * (P_MAX + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(P_MAX)) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, P_MAX + 1, i):
                is_prime_sieve[j] = 0
    primes = [i for i in range(2, P_MAX + 1) if is_prime_sieve[i]]
    
    def find_max_k(p, max_k=200):
        """
        Uses a segmented sieve to find the largest prime factor (LPF) 
        for all numbers in [p^2+1, p^2+max_k].
        Returns the length of the initial run where LPF <= p.
        """
        p2 = p * p
        # lps[i] will store the largest prime factor of p^2 + i + 1
        # Initialize to 0. If 0 at the end, the number is prime (or 1, but n>1 here).
        lps = [0] * max_k
        
        # Sieve with primes q <= p
        # We only care about primes q <= p. If a number has a factor > p, it's not p-smooth.
        # If all factors are <= p, then the largest factor is <= p.
        for q in primes:
            if q > p:
                break
            
            # Find the first multiple of q in [p^2+1, p^2+max_k]
            # We want m such that p^2 + m + 1 is divisible by q
            # m + 1 ≡ -p^2 (mod q)
            # m ≡ -p^2 - 1 (mod q)
            
            # Calculate p^2 mod q
            p_mod = p % q
            p2_mod = (p_mod * p_mod) % q
            # We need (p^2 + 1 + m) % q == 0
            # m ≡ -(p^2 + 1) mod q
            target = (- (p2_mod + 1)) % q
            
            # Find smallest m >= 0 such that m ≡ target (mod q)
            if target == 0:
                # m=0 is a solution? 
                # Check if p^2+1 is divisible by q.
                # If so, start at index 0.
                start = 0
            else:
                start = target
            
            # Ensure start is within [0, max_k-1]
            if start >= max_k:
                continue
                
            for j in range(start, max_k, q):
                lps[j] = q
                
        # Now determine k
        k = 0
        for j in range(max_k):
            n = p2 + j + 1
            lp = lps[j]
            # If lp is 0, the number is prime. 
            # A prime n is p-smooth iff n <= p.
            # But n = p^2 + j + 1 > p^2 >= p (for p>=2).
            # So if lp is 0, n is a prime > p, so it is NOT p-smooth.
            if lp == 0:
                break
            if lp > p:
                break
            k += 1
            
        return k

    best_k = 1
    best_p = 2

    # Iterate through primes in descending order
    # Larger p is generally better for finding long runs
    # We start from the largest primes we have
    for p in reversed(primes):
        if time.time() - start_time > DEADLINE:
            break
            
        # Quick check: p^2+1 must be p-smooth
        # We can just call find_max_k, it's fast enough
        k = find_max_k(p)
        
        if k > best_k:
            best_k = k
            best_p = p
            # If we hit the max possible k (200), we can stop
            if best_k >= 200:
                break

    return {'p': best_p, 'k': best_k}
