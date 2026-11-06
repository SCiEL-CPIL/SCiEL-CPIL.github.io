import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 1000000
    LOW = 10000
    N_LIMIT = N + 1  # We need powerful_part(n+1) for n up to N
    
    # Use NumPy vectorized sieve to compute powerful parts
    # pp[i] = powerful part of i for i in 0..N_LIMIT
    pp = np.ones(N_LIMIT + 1, dtype=np.int64)
    
    # Get primes up to sqrt(N_LIMIT)
    limit = int(math.isqrt(N_LIMIT)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    
    # For each prime p, compute contributions to powerful part
    for p in primes:
        p2 = p * p
        if p2 > N_LIMIT:
            break
        
        # For each power p^k (k>=2), find multiples in range
        power = p2
        k = 2
        while power <= N_LIMIT:
            next_power = power * p
            
            # Multiples of 'power' in [0, N_LIMIT]
            # But we only care about [LOW-1, N_LIMIT] to save time? No, let's do full range.
            # Actually, we can restrict to [1, N_LIMIT]
            if power <= N_LIMIT:
                # Indices: power, 2*power, 3*power, ...
                # We want to multiply pp[idx] by 'power' for these indices
                # But only if the exponent is exactly k, i.e., not divisible by next_power
                # Vectorized approach:
                # Create a boolean mask for multiples of 'power'
                # Then zero out multiples of 'next_power'
                
                # Multiples of power: indices power, 2*power, ...
                # We can use slicing: pp[power::power]
                # But we need to exclude multiples of next_power
                
                # Let's do it in two steps for clarity and safety
                # First, mark all multiples of 'power'
                # Then, unmark (or handle) multiples of 'next_power'
                
                # Actually, the powerful part contribution for prime p in number m is p^e where e is the exact exponent.
                # So if m is divisible by p^k but not p^(k+1), we multiply by p^k.
                
                # Vectorized:
                # Get all indices divisible by 'power'
                idxs = np.arange(power, N_LIMIT + 1, power)
                
                # Filter out those divisible by next_power
                if next_power <= N_LIMIT:
                    mask = (idxs % next_power) != 0
                    idxs = idxs[mask]
                
                # Multiply pp at these indices by 'power'
                if len(idxs) > 0:
                    pp[idxs] *= power
            
            power = next_power
            k += 1
            
            # Check deadline
            if time.time() > deadline - 5.0:
                break
        
        if time.time() > deadline - 5.0:
            break
    
    # Now find the best n in [LOW, N] maximizing pp[n] * pp[n+1]
    # pp is 0-indexed, so pp[i] is powerful part of i
    # We need n in [10000, 1000000], so i in [10000, 1000000]
    # Product is pp[i] * pp[i+1]
    
    if time.time() > deadline - 3.0:
        # Fallback: return a reasonable default
        return LOW
    
    # Vectorized product computation
    # products[i] = pp[i] * pp[i+1] for i in [LOW-1, N-1] (0-indexed)
    # Actually, n ranges from LOW to N, so i ranges from LOW to N
    # pp[i] * pp[i+1]
    
    # Slice pp to get the relevant range
    # i from LOW to N, so pp[LOW:N+1] and pp[LOW+1:N+2]
    # But N+1 might be out of bounds? No, pp has size N_LIMIT+1 = N+2, so indices 0..N+1 are valid
    
    left = pp[LOW:N+1]
    right = pp[LOW+1:N+2]
    
    products = left * right
    
    # Find the index of the maximum product
    max_idx = np.argmax(products)
    best_n = LOW + max_idx
    
    # Check deadline
    if time.time() > deadline - 2.0:
        return int(best_n)
    
    return int(best_n)
