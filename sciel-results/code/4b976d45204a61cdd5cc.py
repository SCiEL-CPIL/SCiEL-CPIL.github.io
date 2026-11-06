"""Find the longest interval in [1,1008] with no modular inverse pairs mod 1009."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    p = 1009
    n = 1008  # elements are 1..1008
    
    # Precompute modular inverses: inv[x] = x^{-1} mod 1009 for x in 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 1
    best_length = 0
    
    # Two-pointer: for each start s, extend end e as far as possible
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
        for e in range(s, n + 1):
            ix = inv[e]
            if ix == e:
                # Self-inverse: e cannot be in any valid interval
                break
            if ix >= s and ix < e:
                # inv[e] is in [s, e-1], so (e, inv[e]) is a conflict pair
                break
            # e is safe to include in [s, e]
            length = e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
        # Early termination: if remaining possible length can't beat best
        if n - s + 1 <= best_length:
            break
    
    if best_length <= 0:
        # Fallback: find any valid single element
        for x in range(1, n + 1):
            if x != 1 and x != 1008:
                best_start = x
                best_length = 1
                break
    
    return [best_start, best_length]
