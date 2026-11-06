import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Primes p ≡ 3 (mod 4)
    primes = [3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83]
    
    # We want to minimize n. So check smaller primes first.
    # But n must be >= 5.
    for p in primes:
        if p < 5:
            continue
            
        if time.time() > deadline:
            break
            
        # Construct Paley tournament on p vertices
        # i -> j iff (j - i) % p is a quadratic residue mod p
        squares = {x * x % p for x in range(1, p)}
        
        # Build adjacency masks
        # mask[i] is a bitmask where bit j is 1 if i -> j
        masks = [0] * p
        for i in range(p):
            m = 0
            for j in range(p):
                if i != j:
                    diff = (j - i) % p
                    if diff in squares:
                        m |= (1 << j)
            masks[i] = m
            
        # Check validity
        # Every 4-set must have a common dominator
        # common dominator u for {i,j,k,l} means u->i, u->j, u->k, u->l
        # i.e., u is in intersection of N+(i), N+(j), N+(k), N+(l)
        # Since u cannot be in {i,j,k,l}, and masks[i] has 0 at i, the intersection
        # will naturally exclude i,j,k,l.
        
        valid = True
        # Iterate over all 4-combinations
        # To speed up, we can break early if invalid
        for i, j, k, l in itertools.combinations(range(p), 4):
            common = masks[i] & masks[j] & masks[k] & masks[l]
            if common == 0:
                valid = False
                break
                
        if valid:
            # Construct the orientation list for the interface
            # one bit per pair (i,j), i<j, lexicographic order
            # 0 means i -> j, 1 means j -> i
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    # Check if i -> j
                    # In Paley, i -> j iff (j - i) % p in squares
                    # We already have masks. masks[i] has bit j set if i -> j
                    if (masks[i] >> j) & 1:
                        orientation.append(0)
                    else:
                        orientation.append(1)
                        
            return {'n': p, 'orientation': orientation}
            
    # Fallback: Return the seed solution (Paley 83) if loop finishes or fails
    # Although the loop includes 83, if 83 is invalid (it shouldn't be), we need a fallback.
    # But 83 is the seed, so it must be valid.
    # If no valid Paley found, return 83 explicitly.
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
