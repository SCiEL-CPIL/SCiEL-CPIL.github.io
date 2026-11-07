"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Sieve to get primes up to 256
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, 257) if is_prime[i]]
    
    # Map each prime to an index for bitmasking
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factors bitmask for each number 2..256
    # If a number has a prime factor > 256 (impossible here), it would be invalid.
    # We only consider numbers whose prime factors are all in 'primes'.
    # All numbers <= 256 have prime factors <= 256.
    
    masks = [0] * 257
    for x in range(2, 257):
        m = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                m |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # temp is a prime
            if temp in prime_to_idx:
                m |= (1 << prime_to_idx[temp])
            else:
                # This shouldn't happen for x <= 256
                pass
        masks[x] = m
        
    # Initial state: all primes are available
    full_mask = (1 << num_primes) - 1
    remaining = full_mask
    
    chosen = [1]
    total_sum = 1
    
    while remaining:
        best_val = 0
        best_mask = 0
        best_x = 0
        
        # Find the number x <= 256 with max value such that masks[x] is a subset of remaining
        # masks[x] & ~remaining == 0
        for x in range(256, 1, -1):
            if masks[x] & ~remaining == 0:
                if x > best_val:
                    best_val = x
                    best_mask = masks[x]
                    best_x = x
                    break # Since we iterate downwards, the first one is the max
        
        if best_x == 0:
            # This should not happen if logic is correct, as primes themselves are valid
            break
            
        chosen.append(best_x)
        total_sum += best_val
        remaining &= ~best_mask
        
    return chosen

# EVOLVE-BLOCK-END
