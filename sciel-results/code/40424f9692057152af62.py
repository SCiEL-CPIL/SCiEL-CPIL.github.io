"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """Return an integer n in [10000, 1000000] maximizing powerful_part(n)*powerful_part(n+1)."""
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 1000000
    low = 10000
    max_val = N + 1
    
    # powerful_part[i] for i in range [1, N+1]
    pp = [1] * (max_val + 1)
    
    # Sieve of Eratosthenes to get primes up to sqrt(max_val)
    limit = int(math.isqrt(max_val)) + 1
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For each prime p, update the powerful part for its multiples
    for p in primes:
        p2 = p * p
        if p2 > max_val:
            break
            
        # Precompute powers of p
        # p_powers[0] = p^0 = 1
        # p_powers[1] = p^1 = p
        # p_powers[2] = p^2 = p2
        # ...
        p_powers = [1, p]
        power = p2
        while power <= max_val:
            p_powers.append(power)
            power *= p
            
        # Iterate through multiples of p^2
        # m = k * p^2
        # v_p(m) = 2 + v_p(k)
        # We need to multiply pp[m] by p^(2 + v_p(k))
        
        # We can iterate k from 1 to max_val // p2
        # For each k, m = k * p2
        # We need v_p(k)
        
        # Optimization: Instead of calculating v_p(k) from scratch for each k,
        # we can use the property that v_p(k*p) = v_p(k) + 1.
        # But k is arbitrary, so we still need to calculate v_p(k).
        
        # Let's stick to the direct calculation but optimized.
        # For k in 1..max_val//p2:
        #   if k % p == 0:
        #      extra_e = 0
        #      x = k
        #      while x % p == 0:
        #         x //= p
        #         extra_e += 1
        #      pp[k*p2] *= p_powers[2 + extra_e]
        #   else:
        #      pp[k*p2] *= p2
        
        # This is O(N/p^2 * log_p(N)) which is manageable.
        
        max_k = max_val // p2
        for k in range(1, max_k + 1):
            m = k * p2
            if k % p == 0:
                x = k
                extra_e = 0
                while x % p == 0:
                    x //= p
                    extra_e += 1
                # Total exponent is 2 + extra_e
                pp[m] *= p_powers[2 + extra_e]
            else:
                # Total exponent is 2
                pp[m] *= p2
                
        # Time check
        if time.time() > deadline - 5:
            break
    
    # Find n in [low, N] that maximizes pp[n] * pp[n+1]
    best_n = low
    best_score = pp[low] * pp[low + 1]
    
    for n in range(low, N):
        sc = pp[n] * pp[n + 1]
        if sc > best_score:
            best_score = sc
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
