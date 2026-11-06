"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 1000000
    powerful_part = np.ones(N + 1, dtype=np.int64)
    
    # Generate primes up to sqrt(N)
    limit = int(np.sqrt(N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For each prime p, process all powers p^e (e>=2)
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        q = p2
        while q <= N:
            next_q = q * p
            if next_q > N:
                # All multiples of q have exact exponent e
                powerful_part[q::q] *= q
            else:
                # Multiples of q but not of next_q
                indices = np.arange(q, N + 1, q)
                mask = (indices % next_q) != 0
                if np.any(mask):
                    powerful_part[indices[mask]] *= q
            q = next_q
    
    # Compute products of consecutive powerful parts
    products = powerful_part[10000:N] * powerful_part[10001:N+1]
    max_idx = np.argmax(products) + 10000
    
    return int(max_idx)

# EVOLVE-BLOCK-END
