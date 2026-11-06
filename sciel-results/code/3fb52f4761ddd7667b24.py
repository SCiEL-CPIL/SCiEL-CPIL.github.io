import time

def construct():
    deadline = time.time() + 85
    TARGET = 123457
    
    # Compute largest prime factor for all integers up to TARGET
    # Using a segmented approach: for each prime p, mark multiples with p as their largest prime factor
    # Actually, we want the LARGEST prime factor, so we process primes in increasing order
    # and overwrite - the last prime that divides n will be the largest
    
    # First, get all primes up to TARGET using sieve
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(TARGET**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, TARGET+1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # Compute largest prime factor for each number
    # Initialize with 1 (for the number 1)
    lpf = [1] * (TARGET + 1)
    for i in range(2, TARGET + 1):
        lpf[i] = i  # initially assume the number itself (will be refined)
    
    # For each prime, mark its multiples
    # Process in increasing order so that larger primes overwrite
    for p in primes:
        if p > TARGET:
            break
        for multiple in range(p, TARGET + 1, p):
            lpf[multiple] = p  # p is a prime factor; since we go in increasing order,
                               # the last assignment gives the largest prime factor
    
    # Now lpf[n] is the largest prime factor of n
    # Verify: for n=1, lpf[1]=1. For primes, lpf[p]=p. For composites, it's the largest prime factor.
    
    # Now find a, b with a+b=TARGET minimizing max(lpf[a], lpf[b])
    # Linear scan: for a from 1 to TARGET-1, b = TARGET - a
    # Track the minimum of max(lpf[a], lpf[b])
    
    best_val = float('inf')
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET):
        b = TARGET - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            if best_val == 2:
                # Can't do better than 2 (since P(1)=1, but we need positive integers,
                # and if both are powers of 2, max would be 2)
                break
    
    if time.time() > deadline - 5:
        # Safety: return best found so far
        return [best_a, best_b]
    
    return [best_a, best_b]
