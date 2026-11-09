import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Generate primes up to 1000 (sqrt(1000001) ~ 1000)
    def sieve(limit):
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(1000)
    # Also keep small primes for quick divisibility checks
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lpf = 1
        # Quick check with small primes first
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                lpf = p
                while n % p == 0:
                    n //= p
        # Continue with remaining primes
        for p in primes:
            if p <= 97:
                continue
            if p * p > n:
                break
            if n % p == 0:
                lpf = p
                while n % p == 0:
                    n //= p
        if n > 1:
            lpf = n
        return lpf
    
    best_n = 10000
    best_val = max(largest_prime_factor(10000), largest_prime_factor(10001))
    
    # Use a faster approach: precompute LPF for all numbers up to 1000001
    # This is much faster than computing on-the-fly
    MAX_N = 1000001
    lpf_array = [1] * (MAX_N + 1)
    
    # Initialize lpf_array with the number itself
    for i in range(2, MAX_N + 1):
        lpf_array[i] = i
    
    # Sieve-like approach to find largest prime factor
    for p in primes:
        if p > MAX_N:
            break
        for multiple in range(p, MAX_N + 1, p):
            lpf_array[multiple] = p
    
    # Now lpf_array[i] contains the largest prime factor of i
    # Find the n that minimizes max(lpf(n), lpf(n+1))
    
    for n in range(10000, MAX_N):
        if time.time() > deadline:
            break
        val = lpf_array[n] if lpf_array[n] > lpf_array[n+1] else lpf_array[n+1]
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
