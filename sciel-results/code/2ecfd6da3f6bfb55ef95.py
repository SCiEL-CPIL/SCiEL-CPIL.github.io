import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N_MAX_VAL = 1000000
    N_MIN_VAL = 10000
    
    # 1. Get primes up to sqrt(10^6) = 1000
    limit = 1000
    sieve = [True] * (limit + 1)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    
    # 2. Initialize powerful part array
    pp = [1] * (N_MAX_VAL + 1)
    
    # 3. Sieve to compute powerful parts
    # For each prime p, we want to multiply pp[m] by p^e where e >= 2
    # if p^e is the exact power of p dividing m.
    # We iterate over powers p^2, p^3, ...
    # For a given power p^k, we iterate over multiples of p^k that are NOT
    # multiples of p^(k+1). For those, the exponent of p is exactly k.
    
    for p in primes:
        p_k = p * p
        k = 2
        while p_k <= N_MAX_VAL:
            # p^k is the current power
            # We need to mark multiples of p^k that are not multiples of p^(k+1)
            # p^(k+1) = p * p^k
            next_p_k = p_k * p
            
            # Iterate i such that m = i * p_k
            # If i is not divisible by p, then m is not divisible by p^(k+1)
            # So v_p(m) = k exactly.
            # We multiply pp[m] by p^k
            
            # Optimization:
            # Multiples of p_k are p_k, 2*p_k, 3*p_k, ...
            # next_p_k is a multiple of p_k (specifically p * p_k).
            # So the indices i that are multiples of p correspond to m that are multiples of p^(k+1).
            
            p_pow_k = p ** k
            
            i = 1
            m = p_k
            while m <= N_MAX_VAL:
                if i % p != 0:
                    pp[m] *= p_pow_k
                i += 1
                m += p_k
            
            p_k = next_p_k
            k += 1
            
            if time.time() > deadline - 5:
                break
    
    if time.time() > deadline - 5:
        return N_MIN_VAL

    # 4. Find the best n
    best_n = N_MIN_VAL
    best_score = 0
    
    for n in range(N_MIN_VAL, N_MAX_VAL):
        score = pp[n] * pp[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
        if time.time() > deadline - 2:
            break

    return best_n
