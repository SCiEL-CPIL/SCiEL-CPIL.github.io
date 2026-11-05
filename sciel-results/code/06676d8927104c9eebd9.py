import math
import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n.
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin

    # Precompute largest prime factor for all numbers 1..TARGET using a sieve.
    # lp[i] = largest prime factor of i.
    lp = [0] * (TARGET + 1)
    lp[1] = 1
    for i in range(2, TARGET + 1):
        if lp[i] == 0:
            # i is prime
            for j in range(i, TARGET + 1, i):
                if lp[j] == 0:
                    lp[j] = j
                else:
                    lp[j] = i  # i is the largest prime factor seen so far for j (since we go in increasing order)
            # Wait, that's not right. Let me think again.
            # Better approach: for each prime p, mark multiples.
            # Actually, a cleaner sieve for largest prime factor:
    
    # Redo: proper sieve for largest prime factor
    lp = [0] * (TARGET + 1)
    lp[1] = 1
    for i in range(2, TARGET + 1):
        if lp[i] == 0:
            # i is prime
            for j in range(i, TARGET + 1, i):
                lp[j] = i  # since i increases, the last prime that divides j is the largest
    
    # Now lp[n] is the largest prime factor of n for all n in 1..TARGET.
    # Scan all a from 1 to TARGET-1, compute max(lp[a], lp[TARGET-a]), find minimum.
    best_val = float('inf')
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET):
        b = TARGET - a
        m = max(lp[a], lp[b])
        if m < best_val:
            best_val = m
            best_a = a
            best_b = b
        if time.time() - start_time > DEADLINE - 2:
            break
    
    return [best_a, best_b]
