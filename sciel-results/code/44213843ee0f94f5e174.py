import time

def construct():
    deadline = time.time() + 85
    TARGET = 123457
    
    # Compute the largest prime factor for every integer from 1 to TARGET
    # using a modified sieve approach
    lpf = [0] * (TARGET + 1)  # largest prime factor
    
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lpf[j] = i  # i is a prime factor of j, and since we go in increasing order, it's the largest
    
    # lpf[1] = 1 by definition (P(1) = 1)
    lpf[1] = 1
    
    # Now scan all pairs (a, b) where a + b = TARGET, a >= 1, b >= 1
    # We want to minimize max(lpf[a], lpf[b])
    best_pair = None
    best_M = float('inf')
    
    # Optimization: since a + b = TARGET and we want to minimize max(P(a), P(b)),
    # we can iterate and track the minimum possible max value.
    # For efficiency, we can sort by potential and break early if possible.
    # But a full scan of ~123456 pairs is fast enough.
    
    for a in range(1, TARGET):
        if time.time() > deadline - 2:
            break
        b = TARGET - a
        M = max(lpf[a], lpf[b])
        if M < best_M:
            best_M = M
            best_pair = [a, b]
            # If best_M is 1, that's the absolute minimum possible (both are 1),
            # but that would require a=b=1 which sums to 2, not TARGET.
            # The next best is M=2, meaning both a and b are powers of 2.
            # We can't easily break early without more structure, so continue.
    
    if best_pair is not None:
        return best_pair
    
    # Fallback (should never happen)
    return [1, TARGET]
