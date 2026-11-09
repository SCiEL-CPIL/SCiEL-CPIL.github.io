import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # We need to find n in [10000, 1000000] that minimizes largest_prime_factor(n * (n+1))
    # Note: Since gcd(n, n+1) = 1, the prime factors of n(n+1) are the union of prime factors of n and n+1.
    # Thus, LPF(n(n+1)) = max(LPF(n), LPF(n+1)).
    
    # Strategy:
    # 1. Precompute smallest prime factor (SPF) for all numbers up to 1000001 using a sieve.
    # 2. Use SPF to quickly compute the largest prime factor of any number in range.
    # 3. Scan all n in [10000, 1000000] and find the one with the minimum max(LPF(n), LPF(n+1)).
    
    MAX_N = 1000000
    LIMIT = MAX_N + 1  # We need to factor up to 1000001
    
    # Sieve for Smallest Prime Factor
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Function to get largest prime factor using SPF
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lpf = 1
        while n > 1:
            p = spf[n]
            lpf = p
            while n % p == 0:
                n //= p
        return lpf
    
    best_n = 10000
    best_lpf = float('inf')
    
    # Iterate through all candidates
    for n in range(10000, MAX_N + 1):
        if time.time() > deadline:
            break
            
        lpf_n = largest_prime_factor(n)
        lpf_n1 = largest_prime_factor(n + 1)
        current_lpf = max(lpf_n, lpf_n1)
        
        if current_lpf < best_lpf:
            best_lpf = current_lpf
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
