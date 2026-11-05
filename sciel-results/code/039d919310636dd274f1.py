"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 1000000
    # Sieve of Eratosthenes to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Compute powerful part P(n) for all n in [1, N]
    # P(n) = product of p^e for all prime powers p^e dividing n with e >= 2
    # i.e., the largest powerful divisor of n
    P = [1] * (N + 1)
    
    for p in primes:
        # For each prime p, compute p^2, p^3, ... <= N
        pk = p * p
        while pk <= N:
            # For every multiple m of pk, p^(exponent of p in m) contributes to P(m)
            # We need to multiply P[m] by p^(e) where e is the full exponent of p in m
            # But we can do this more efficiently:
            # For each multiple m of pk, we need to find the highest power of p dividing m
            # and multiply P[m] by that power.
            
            # Iterate over multiples of pk
            for m in range(pk, N + 1, pk):
                # Find the full exponent of p in m
                # Start with pk = p^2, check if m is divisible by higher powers
                temp = m
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                if exp >= 2:
                    P[m] *= (p ** exp)
            pk *= p
    
    # Now find n in [10000, 1000000] that maximizes P[n] * P[n+1]
    best_n = 10000
    best_val = P[10000] * P[10001]
    
    for n in range(10000, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
