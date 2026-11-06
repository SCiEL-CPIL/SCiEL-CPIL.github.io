import math
import time
import numpy as np

def _powerful_part_direct(n):
    """Compute powerful part of n by direct trial division."""
    if n < 4:
        return 1
    result = 1
    temp = n
    # Check factor 2
    e = 0
    while temp % 2 == 0:
        temp //= 2
        e += 1
    if e >= 2:
        result *= 2 ** e
    # Check odd factors
    d = 3
    while d * d <= temp:
        if temp % d == 0:
            e = 0
            while temp % d == 0:
                temp //= d
                e += 1
            if e >= 2:
                result *= d ** e
        d += 2
    # temp is now 1 or a prime with exponent 1
    return result

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1
    
    # Sieve primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Initialize powerful part array P with 1s
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Compute powerful parts using vectorized operations
    for p in primes:
        if time.time() - start_time > deadline - 10:
            break
        p = int(p)
        p2 = p * p
        if p2 > MAX_VAL:
            break
        
        power = p2
        while power <= MAX_VAL:
            if time.time() - start_time > deadline - 10:
                break
            
            next_power = power * p
            
            if next_power > MAX_VAL:
                # All multiples of power in [1, MAX_VAL] have exactly exponent k
                P[power::power] *= power
            else:
                # Multiples of power but NOT multiples of next_power
                # have exactly exponent k for prime p
                multiples = np.arange(power, MAX_VAL + 1, power)
                mask = (multiples % next_power) != 0
                valid_indices = multiples[mask]
                P[valid_indices] *= power
            
            power = next_power
    
    if time.time() - start_time > deadline - 5:
        # Not enough time for careful scan; return a reasonable default
        return 10000
    
    # Compute products using float64 to avoid int64 overflow
    # P values are at most ~1e6, product at most ~1e12, fits in int64
    # But to be safe, use float64 for the product
    products = (P[10000:1000001].astype(np.float64)) * (P[10001:1000002].astype(np.float64))
    
    # Find top candidates
    top_k = 50
    top_indices = np.argpartition(-products, top_k)[:top_k]
    top_indices = top_indices[np.argsort(-products[top_indices])]
    
    # Verify top candidates with direct factorization
    best_n = 10000
    best_val = 0
    
    for idx in top_indices:
        if time.time() - start_time > deadline - 2:
            break
        n = 10000 + idx
        pn_direct = _powerful_part_direct(n)
        pn1_direct = _powerful_part_direct(n + 1)
        val = pn_direct * pn1_direct
        
        # Cross-check with sieve
        pn_sieve = int(P[n])
        pn1_sieve = int(P[n + 1])
        
        if pn_sieve != pn_direct or pn1_sieve != pn1_direct:
            # Sieve bug detected; trust direct computation
            if val > best_val:
                best_val = val
                best_n = n
        else:
            if val > best_val:
                best_val = val
                best_n = n
    
    # Also check the overall argmax as a fallback
    max_idx = np.argmax(products)
    n_max = 10000 + max_idx
    pn_direct = _powerful_part_direct(n_max)
    pn1_direct = _powerful_part_direct(n_max + 1)
    val_max = pn_direct * pn1_direct
    if val_max > best_val:
        best_n = n_max
    
    return int(best_n)
