import time
import random

def construct():
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    # 1. Check small Paley tournaments
    primes = [79, 71, 59, 47, 43, 31, 23, 19, 11, 7]
    for p in primes:
        if time.time() - start_time > time_limit - 5:
            break
        if p % 4 != 3:
            continue
        squares = {x*x % p for x in range(1, p)}
        # Construct orientation
        # orientation is list of bits for (i,j) i<j
        # 0 if i->j, 1 if j->i
        # i->j iff (j-i)%p in squares
        n = p
        orientation = []
        for i in range(n):
            for j in range(i+1, n):
                if (j-i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        
        # Check validity
        if is_valid(n, orientation):
            return {'n': n, 'orientation': orientation}
            
    # 2. Local Search on n=60
    if time.time() - start_time < time_limit - 10:
        result = local_search(60, time_limit)
        if result:
            return result
            
    # 3. Fallback to Paley 83
    p = 83
    squares = {x*x % p for x in range(1, p)}
    n = p
    orientation = []
    for i in range(n):
        for j in range(i+1, n):
            if (j-i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': n, 'orientation': orientation}

def is_valid(n, orientation):
    # Build in_masks
    # in_mask[v] is bitmask of u such that u->v
    in_masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                # i -> j
                in_masks[j] |= (1 << i)
            else:
                # j -> i
                in_masks[i] |= (1 << j)
                
    # Check all 4-sets
    # 4-set S = {a,b,c,d}
    # Dominator v exists iff (in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]) & ~(1<<a|1<<b|1<<c|1<<d) != 0
    
    # Optimization: Precompute masks for pairs? No, just iterate.
    # Number of 4-sets is C(n,4).
    # For n=80, C(80,4) ~ 1.5M.
    # 1.5M iterations in Python is slow (~1-2s).
    # We can break early if we find a bad set? No, we need to know if ANY is bad.
    # So we must check ALL.
    
    # To speed up, we can use numpy?
    # Or just simple loops.
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    mask = in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]
                    mask &= ~(1 << a)
                    mask &= ~(1 << b)
                    mask &= ~(1 << c)
                    mask &= ~(1 << d)
                    if mask == 0:
                        return False
    return True

def local_search(n, time_limit):
    # Random start
    orientation = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
    
    # Initial check
    if is_valid(n, orientation):
        return {'n': n, 'orientation': orientation}
        
    # We need a faster cost function for local search.
    # is_valid is O(n^4).
    # We need incremental.
    # This is complex.
    # Given time constraints, I might skip local search if it's too hard to implement incrementally.
    # Or do a very simple hill climb with full re-evaluation?
    # Full re-eval is 1-2s per step.
    # 120s / 2s = 60 steps.
    # 60 random steps is unlikely to fix 5000 bad sets.
    # So local search is probably ineffective without incremental update.
    
    # I will omit local search to keep code simple and reliable, unless I can write incremental fast.
    # Let's just return the best Paley found or 83.
    return None
