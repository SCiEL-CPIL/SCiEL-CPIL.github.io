import time
import math

def construct():
    start_time = time.time()
    deadline = 58 # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Compute largest prime factor for all numbers up to N_MAX+1
    # Using a modified sieve: for each prime p, mark multiples of p with p as their largest prime factor
    max_val = N_MAX + 1
    lpf = [1] * (max_val + 1)  # largest prime factor
    
    # Sieve for primes and compute largest prime factor
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            # i is prime
            for j in range(i, max_val + 1, i):
                is_prime[j] = False
                lpf[j] = i  # i is a prime factor; we want the LARGEST, so we keep updating
            # Note: since we iterate i in increasing order, later (larger) primes will overwrite
            # Actually, we need the LARGEST prime factor. Let's think differently.
            # For each number, its largest prime factor is the largest prime that divides it.
            # We can do: for each prime p, for each multiple m of p, set lpf[m] = max(lpf[m], p)
            # But that's expensive. Alternative: after finding all primes, for each number,
            # its largest prime factor is the largest prime p such that p divides the number.
            # We can compute this by: lpf[n] = n if n is prime, else lpf[n] = max(p for p in primes if n%p==0)
            # But that's O(pi(n) * n) which is too slow.
            # Better: use the fact that if n = p * k where p is the smallest prime factor,
            # then lpf[n] = max(p, lpf[k]). But we want the LARGEST, not smallest.
            # Actually, the largest prime factor of n is either n itself (if prime) or
            # the largest prime factor of n/p where p is any prime factor.
            # We can use: lpf[n] = max(p, lpf[n//p]) for any prime p dividing n.
            # But we need to be careful. Let's use a different approach:
            # Initialize lpf[n] = n for all n. Then for each prime p, for each multiple m of p,
            # we know p divides m, so lpf[m] should be at least p. But we want the MAX.
            # So if we process primes in INCREASING order, and set lpf[m] = p for multiples,
            # the LAST prime that divides m will be the largest. That's exactly what we want!
            # So the code above is correct: for each prime i (in increasing order),
            # for each multiple j of i, set lpf[j] = i. The last assignment wins, which is the largest prime factor.
    
    # Now scan for the best n
    best_n = N_MIN
    best_lpf = float('inf')
    
    for n in range(N_MIN, N_MAX + 1):
        # LPF of n(n+1) = max(lpf[n], lpf[n+1])
        current_lpf = max(lpf[n], lpf[n + 1])
        if current_lpf < best_lpf:
            best_lpf = current_lpf
            best_n = n
        
        # Time check
        if time.time() - start_time > deadline:
            break
    
    return best_n
