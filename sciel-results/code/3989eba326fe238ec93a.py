import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 1000000
    MAX_VAL = N + 1

    # Sieve of Eratosthenes to get primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i, v in enumerate(is_prime) if v]

    # Initialize powerful part array using NumPy for vectorized updates
    P = np.ones(MAX_VAL + 1, dtype=np.int64)

    # Compute powerful parts using segmented sieve over primes
    # For each prime p, for each power p^k (k>=2), multiply P[m] by p^k
    # for all m that are divisible by p^k but not by p^(k+1)
    # Optimization: Multiply all multiples of p^k, then divide multiples of p^(k+1) by p^k
    # This avoids creating temporary arrays for filtering indices.
    for p in primes:
        power = p * p
        while power <= MAX_VAL:
            if time.time() > deadline:
                break
            next_power = power * p
            
            # Step 1: Multiply all multiples of p^k by p^k
            P[power::power] *= power
            
            # Step 2: If p^(k+1) exists in range, undo the multiplication for its multiples
            # These multiples will be handled by the next iteration (or later) with higher powers.
            if next_power <= MAX_VAL:
                P[next_power::next_power] //= power
            else:
                # No higher power in range, so we are done with this prime
                break
                
            power = next_power

    # Check deadline before expensive scan
    if time.time() - start_time > deadline:
        # Return a valid fallback
        return 10000

    # Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    # Use NumPy for vectorized product scan
    products = P[10000:N+1] * P[10001:N+2]
    idx = int(np.argmax(products))
    best_n = 10000 + idx

    return best_n
