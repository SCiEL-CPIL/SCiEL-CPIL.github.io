import time
import math

def sieve(limit):
    """Generate primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
    return [i for i, val in enumerate(is_prime) if val]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate primes up to 10^7
    # 10^7 primes sieve is fast (~0.5s)
    limit = 10**7
    primes = sieve(limit)
    
    best_k = 0
    best_p = 2
    
    # Pre-compute small primes for quick access? 
    # The main loop iterates primes.
    
    for p in primes:
        if time.time() > deadline:
            break
        
        p_sq = p * p
        # We need to check p^2 + 1 ... p^2 + 200
        # Initialize values
        vals = [p_sq + i for i in range(1, 201)]
        
        # Iterate through primes q <= p
        # We can break early if q > p
        for q in primes:
            if q > p:
                break
            
            # Find first multiple of q in [p_sq + 1, p_sq + 200]
            # m * q >= p_sq + 1
            # m = ceil((p_sq + 1) / q)
            
            # Optimization: if q > 200, at most one multiple
            if q > 200:
                # Calculate first multiple
                # (p_sq + 1) % q
                rem = (p_sq + 1) % q
                if rem == 0:
                    first_mult = p_sq + 1
                else:
                    first_mult = p_sq + 1 + (q - rem)
                
                if first_mult <= p_sq + 200:
                    idx = first_mult - (p_sq + 1)
                    # Divide out q
                    while vals[idx] % q == 0:
                        vals[idx] //= q
            else:
                # Standard sieve step
                # First multiple >= p_sq + 1
                rem = (p_sq + 1) % q
                if rem == 0:
                    v = p_sq + 1
                else:
                    v = p_sq + 1 + (q - rem)
                
                while v <= p_sq + 200:
                    idx = v - (p_sq + 1)
                    while vals[idx] % q == 0:
                        vals[idx] //= q
                    v += q
        
        # Count how many vals are 1
        k = 0
        for v in vals:
            if v == 1:
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
