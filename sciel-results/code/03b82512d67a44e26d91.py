import itertools
import time

def construct():
    # Check Paley tournaments for primes p = 3 mod 4
    # We want the smallest n, so we check in increasing order.
    # p must be a prime and p % 4 == 3.
    # Also n >= 5.
    candidates = [7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83]
    
    for p in candidates:
        if p < 5:
            continue
        if p % 4 != 3:
            continue
            
        # Construct Paley tournament
        squares = {x * x % p for x in range(1, p)}
        
        # Check validity using bitmasks
        # In[v] is a bitmask of vertices u such that u -> v
        In = [0] * p
        for u in range(p):
            mask = 0
            for v in range(p):
                if v == u:
                    continue
                # v -> u if (u - v) % p is in squares
                # Note: i -> j iff (j-i) in squares.
                # So v -> u iff (u-v) in squares.
                if (u - v) % p in squares:
                    mask |= (1 << v)
            In[u] = mask
            
        # Check all 4-sets
        valid = True
        for S in itertools.combinations(range(p), 4):
            inter = In[S[0]]
            for u in S[1:]:
                inter &= In[u]
                if inter == 0:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            # Construct the orientation list
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    # i -> j if (j - i) % p in squares
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
            
    # Fallback to 83 if something went wrong (should not happen)
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
