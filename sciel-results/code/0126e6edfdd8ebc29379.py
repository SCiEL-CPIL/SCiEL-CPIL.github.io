"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return integer n in [10000, 1000000] minimizing max(lpf(n), lpf(n+1)).
    Uses a segmented sieve approach to compute largest prime factors efficiently.
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000  # n can be up to 1000000, so n+1 up to 1000001
    
    # We need to compute LPF for all numbers from N_MIN to N_MAX+1
    # Then find n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    
    # Strategy: 
    # 1. Find all primes up to sqrt(N_MAX+1) ~ 1000
    # 2. For each number in [N_MIN, N_MAX+1], divide out small primes and track largest factor
    # 3. Any remaining factor > 1 is prime (since we divided out all primes up to sqrt)
    
    # Actually, let's think more carefully.
    # LPF(x) = largest prime factor of x.
    # For x in [10000, 1000001], sqrt(1000001) ≈ 1000.
    # So we need primes up to 1000 to factor any number in this range.
    # After dividing out all primes p <= sqrt(x), the remaining factor is either 1 or prime.
    
    # Step 1: Generate primes up to 1000
    limit = 1001
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    small_primes = [p for p in range(2, limit + 1) if is_prime[p]]
    
    # Step 2: Compute LPF for each number in [N_MIN, N_MAX+1]
    # We'll use an array where lpf[i] stores the largest prime factor of (N_MIN + i)
    # Actually, let's store for each number directly.
    
    # For efficiency, we process by dividing each number by small primes.
    # But that's O(N * pi(sqrt(N))) which is 10^6 * 168 ≈ 1.68 * 10^8 operations.
    # That might be too slow in Python.
    
    # Better approach: Sieve-like method.
    # Initialize lpf array with 0.
    # For each small prime p, for each multiple m of p in [N_MIN, N_MAX+1],
    #   set lpf[m] = max(lpf[m], p)
    # After processing all small primes, for any number x where lpf[x] < x and x > 1,
    #   the remaining factor (x / product of all prime factors found) is a prime > sqrt(x),
    #   so we need to handle that.
    
    # Actually, a cleaner approach:
    # For each number x in [N_MIN, N_MAX+1]:
    #   temp = x
    #   lpf = 1
    #   for p in small_primes:
    #       if p * p > temp: break
    #       if temp % p == 0:
    #           lpf = p
    #           while temp % p == 0:
    #               temp //= p
    #   if temp > 1:
    #       lpf = max(lpf, temp)
    #   store lpf[x] = lpf
    
    # This is O(N * pi(sqrt(N))) in worst case, but typically much faster.
    # Let's estimate: average number of divisions is small.
    # For 10^6 numbers, this should run in a few seconds in Python.
    
    n_values = N_MAX + 1 - N_MIN + 1  # count of numbers from N_MIN to N_MAX+1
    lpf = [0] * (n_values + 1)  # lpf[i] corresponds to N_MIN + i
    
    for i in range(n_values):
        x = N_MIN + i
        temp = x
        lpf_val = 1
        for p in small_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                lpf_val = p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            lpf_val = max(lpf_val, temp)
        lpf[i] = lpf_val
    
    # Now find n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    # lpf[i] is for N_MIN + i, so lpf[n - N_MIN] is lpf(n) and lpf[n+1 - N_MIN] is lpf(n+1)
    
    best_n = N_MIN
    best_score = float('inf')
    
    for n in range(N_MIN, N_MAX + 1):
        idx = n - N_MIN
        score = max(lpf[idx], lpf[idx + 1])
        if score < best_score:
            best_score = score
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
