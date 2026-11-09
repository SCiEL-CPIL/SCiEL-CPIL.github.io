"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 1000000
    
    # Precompute primes up to sqrt(N) for trial division
    limit = int(math.isqrt(N)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def largest_prime_factor(n):
        """Compute largest prime factor of n using trial division by precomputed primes."""
        if n <= 1:
            return 1
        temp = n
        lpf = 1
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                lpf = p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            lpf = temp
        return lpf
    
    min_val = float('inf')
    best_n = 10000
    
    # Search for n in [10000, 999999] such that max(lpf[n], lpf[n+1]) is minimized
    for n in range(10000, N):
        val_n = largest_prime_factor(n)
        val_np1 = largest_prime_factor(n + 1)
        val = max(val_n, val_np1)
        
        if val < min_val:
            min_val = val
            best_n = n
            # Early termination: if we find a very smooth pair, we can stop
            # The theoretical minimum for numbers >= 10000 is at least 2
            # But practically, if min_val is small enough, further improvement is unlikely
            if min_val <= 13:
                break
                
    return best_n
# EVOLVE-BLOCK-END
