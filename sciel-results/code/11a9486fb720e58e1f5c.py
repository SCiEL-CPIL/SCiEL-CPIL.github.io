"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def construct():
    """
    Return two positive integers a, b with a + b = 123457.
    Minimize max(P(a), P(b)) where P(n) is the largest prime factor of n (P(1)=1).
    """
    target = 123457
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Step 1: Compute smallest prime factor (SPF) up to target
    # This allows O(log n) factorization for any number <= target
    spf = list(range(target + 1))
    for i in range(2, int(math.isqrt(target)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, target + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Precompute largest prime factor for all numbers up to target
    # Using SPF: P(n) can be computed by repeatedly dividing by spf[n]
    # and tracking the last (largest) prime factor
    lp = [1] * (target + 1)  # lp[n] = largest prime factor of n
    for n in range(2, target + 1):
        x = n
        last_prime = 1
        while x > 1:
            p = spf[x]
            last_prime = p
            while x % p == 0:
                x //= p
        lp[n] = last_prime
    
    # Step 3: For each a in [1, target-1], b = target - a
    # Compute max(lp[a], lp[b]) and find the minimum
    best_val = float('inf')
    best_a = 1
    best_b = target - 1
    
    for a in range(1, target // 2 + 1):  # symmetry: a <= b
        b = target - a
        val = max(lp[a], lp[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
        if time.time() > deadline:
            break
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
