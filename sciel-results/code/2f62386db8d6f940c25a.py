import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 1000000
    # powerful_part[i] will store the powerful part of i
    pp = [1] * (N + 2)  # index 0..N+1
    
    # Sieve: for each prime, mark contributions to powerful parts
    # For each prime p, for each power p^k where k>=2, 
    # multiply pp[m] by p^(k - previous_k) for multiples m of p^k but not p^(k+1)
    # Actually simpler: for each number, track the max power of each prime dividing it with exponent >= 2
    
    # Better: for each prime p, iterate k=2,3,... while p^k <= N
    # For each such p^k, for each multiple m of p^k, we know p^k divides m.
    # The exponent of p in m is at least k. We want the full power p^e where e is the actual exponent.
    
    # Alternative: compute for each number its prime factorization's powerful part directly.
    # Use smallest prime factor sieve.
    
    # SPF sieve
    spf = list(range(N + 2))
    for i in range(2, int(math.isqrt(N + 1)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    def powerful_part(m):
        """Compute powerful part of m using SPF."""
        if m <= 1:
            return 1
        result = 1
        while m > 1:
            p = spf[m]
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            if e >= 2:
                result *= p ** e
        return result
    
    best_n = 10000
    best_val = powerful_part(10000) * powerful_part(10001)
    
    # We can't check all n from 10000 to 1000000 naively if it's too slow
    # Let's estimate: ~990000 iterations, each with O(log n) factorization
    # That should be fine within 85 seconds.
    
    for n in range(10000, N):
        if time.time() - start_time > DEADLINE:
            break
        v = powerful_part(n) * powerful_part(n + 1)
        if v > best_val:
            best_val = v
            best_n = n
    
    return best_n
