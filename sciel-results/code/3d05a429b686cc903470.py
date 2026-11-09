import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    N = 123457
    
    # Sieve to compute largest prime factor for all numbers up to N
    # lpf[i] will store the largest prime factor of i
    lpf = [1] * (N + 1)
    
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    # Scan all decompositions
    # We can iterate a from 1 to N//2 because (a, b) and (b, a) yield the same max(P(a), P(b))
    # But iterating all is also fast enough. Let's iterate all for simplicity and safety against off-by-one in symmetry logic if any.
    for a in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            # If we find a very small value, we might break early?
            # The minimum possible value for max(P(a), P(b)) is 2 if one is 1 and the other is a power of 2.
            # 123456 is not a power of 2.
            # Next best is 3? If one is 1 and other is 3 * 2^k? 
            # Or if both are composed of small primes.
            # Since N is small, we just finish the loop.
            
    return [best_a, best_b]
