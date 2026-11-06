import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

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

    # Initialize powerful part array
    # Using int64 to hold products up to 10^12
    P = np.ones(MAX_VAL + 1, dtype=np.int64)

    # Compute powerful parts using segmented sieve over primes
    for p in primes:
        power = p * p
        while power <= MAX_VAL:
            if time.time() > deadline:
                # Fallback if timeout is imminent
                # Return a valid number, preferably one with high powerful part
                # 10000 is a safe fallback, but we can try to return something better
                # If we haven't finished, we can't guarantee optimality.
                # However, the sieve is very fast, this should not happen.
                return 10000
            
            next_power = power * p
            
            # Multiply all multiples of p^k by p^k
            P[power::power] *= power
            
            # If p^(k+1) exists, undo the multiplication for its multiples
            # They will be handled by the next power
            if next_power <= MAX_VAL:
                P[next_power::next_power] //= power
            else:
                break
                
            power = next_power

    # Check deadline
    if time.time() > deadline:
        return 10000

    # Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    # Vectorized product scan
    # Ensure we don't go out of bounds
    # P has indices 0..MAX_VAL
    # We want n in 10000..N
    # So P[n] is P[10000..N]
    # P[n+1] is P[10001..N+1]
    
    # N = 1000000
    # P[10000:1000001] has length 990001
    # P[10001:1000002] has length 990001
    
    products = P[10000:N+1] * P[10001:N+2]
    
    if products.size == 0:
        return 10000

    idx = int(np.argmax(products))
    best_n = 10000 + idx
    
    return best_n
