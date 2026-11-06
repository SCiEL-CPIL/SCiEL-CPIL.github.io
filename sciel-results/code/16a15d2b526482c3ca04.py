"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 1000000
    LO = 10000
    HI = 1000000
    
    # Generate primes up to sqrt(N) using simple sieve
    limit = int(np.sqrt(N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # powerful_part for the range [LO, HI]
    # powerful_part[i - LO] = powerful part of i
    size = HI - LO + 1
    powerful_part = np.ones(size, dtype=np.int64)
    
    # For each prime p, process all powers p^e (e >= 2)
    for p in primes:
        p2 = p * p
        if p2 > HI:
            break
        q = p2
        while q <= HI:
            next_q = q * p
            # Find all multiples of q in [LO, HI]
            # First multiple >= LO: ceil(LO/q) * q
            first = ((LO + q - 1) // q) * q
            if first > HI:
                break
            
            # Create array of indices in the local array for multiples of q
            # These are numbers: first, first+q, first+2q, ... <= HI
            # Local index: num - LO
            indices = np.arange(first, HI + 1, q)
            local_indices = indices - LO
            
            if next_q > HI:
                # All these have exact exponent e (no higher power of p fits)
                powerful_part[local_indices] *= q
            else:
                # Filter: keep only those NOT divisible by next_q
                mask = (indices % next_q) != 0
                if np.any(mask):
                    powerful_part[local_indices[mask]] *= q
            
            q = next_q
    
    # Compute product of consecutive powerful parts
    # powerful_part[i] corresponds to number LO + i
    # We want max over i in [0, size-2] of powerful_part[i] * powerful_part[i+1]
    # which corresponds to numbers (LO+i) and (LO+i+1)
    products = powerful_part[:-1] * powerful_part[1:]
    max_local_idx = np.argmax(products)
    result = LO + max_local_idx
    
    return int(result)

# EVOLVE-BLOCK-END
