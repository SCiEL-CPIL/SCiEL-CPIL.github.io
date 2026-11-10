# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    limit = MAX_N + 1  # we need pp up to MAX_N+1 for n+1
    
    # Build list of primes up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Compute powerful part for each number using sieve-like accumulation
    # For each prime p, for each exponent e >= 2, multiply pp[multiple of p^e] by p^e
    pp = [1] * (limit + 1)
    
    for p in primes:
        p2 = p * p
        if p2 > limit:
            break
        # For exponent e = 2, 3, 4, ... while p^e <= limit
        pe = p2
        e = 2
        while pe <= limit:
            # All multiples of pe have p^e as a factor
            # But we only want the FULL p^e if the exponent in n is exactly e
            # Actually, for the powerful part, we want p^e where e is the actual exponent
            # So we need to be careful: if n is divisible by p^e but not p^(e+1),
            # then p^e is part of the powerful part.
            # 
            # Better approach: for each n, determine the exact exponent of p,
            # and if >= 2, multiply by p^exact_exponent.
            # 
            # Sieve approach: for each p, compute the exponent of p in each multiple.
            # For multiples of p^k but not p^(k+1), the exponent is k.
            
            # Let's do it differently: for each p, iterate through powers
            # For p^e, mark numbers divisible by p^e but not p^(e+1)
            # This is complex. Let's use a different strategy.
            
            break  # We'll use a different approach below
    
    # Better approach: for each prime p, for each multiple of p^2, 
    # determine the exact power of p dividing that multiple, and if >= 2,
    # multiply pp[multiple] by p^exact_power.
    # But we can do this more efficiently.
    
    # For each prime p:
    #   For k = 2, 3, 4, ... while p^k <= limit:
    #     For each multiple m of p^k that is NOT a multiple of p^(k+1):
    #       pp[m] *= p^k
    # This is equivalent to: for each m divisible by p^2, find the exact
    # exponent and multiply accordingly.
    
    # Even better: for each prime p, for each multiple m of p^2:
    #   count how many times p divides m (starting from p^2)
    #   pp[m] *= p^count
    
    # Let's use the direct approach but optimized:
    for p in primes:
        p2 = p * p
        if p2 > limit:
            break
        # For each multiple of p^2, determine exact exponent of p
        for m in range(p2, limit + 1, p2):
            # Count exponent of p in m
            temp = m
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp >= 2:
                pp[m] *= p ** exp
    
    # Find n in [MIN_N, MAX_N] that maximizes pp[n] * pp[n+1]
    best_n = MIN_N
    best_score = 0
    
    for n in range(MIN_N, MAX_N):
        score = pp[n] * pp[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
