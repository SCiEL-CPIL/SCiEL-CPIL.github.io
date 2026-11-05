import time
import itertools

def construct():
    start_time = time.time()
    deadline = 120.0 # 120s limit, keep safety margin
    
    # Primes p where p % 4 == 3, in range [31, 90]
    # We check in increasing order to minimize n
    candidates = [31, 43, 47, 59, 67, 71, 79, 83]
    
    best_n = 83 # Fallback to current known valid
    best_orientation = None
    
    # Precompute orientation for 83 to have a fallback
    p = 83
    squares = {x * x % p for x in range(1, p)}
    best_orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    
    for p in candidates:
        if time.time() > deadline - 10: # Safety margin
            break
            
        # Construct Paley tournament
        squares = {x * x % p for x in range(1, p)}
        
        # adj[i][j] is 1 if i -> j
        # We need to check if there exists a 4-set S such that S dominates V\S
        # i.e., for all v not in S, there exists u in S such that u -> v.
        
        # Build adjacency matrix as list of sets or bitsets for fast intersection
        # n is small, so sets are fine.
        out_neighbors = [set() for _ in range(p)]
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                if (j - i) % p in squares:
                    out_neighbors[i].add(j)
        
        # Check validity:
        # Invalid if exists S (size 4) such that union of out_neighbors[u] for u in S covers V\S
        # V\S is V - S.
        # So we check: (union_{u in S} out_neighbors[u]) >= (V - S)
        
        valid = True
        # Optimization: If any vertex v has out-degree < 3? No.
        # Just iterate all 4-sets.
        # For p=31, C(31,4) = 31465. Very small.
        # For p=83, C(83,4) = 1.7M. Still small.
        
        # To speed up, we can use bitsets (Python ints)
        bitsets = [0] * p
        for i in range(p):
            mask = 0
            for j in out_neighbors[i]:
                mask |= (1 << j)
            bitsets[i] = mask
            
        all_mask = (1 << p) - 1
        
        # Iterate all 4-sets
        # We can break early if we find an invalid set
        # An invalid set S is one where (OR of bitsets[u] for u in S) covers all bits except S itself?
        # No, it must cover V\S.
        # Let U_S = OR_{u in S} bitsets[u].
        # We need U_S to contain all v in V\S.
        # This is equivalent to: (U_S | mask_S) == all_mask, where mask_S is the bitmask of S.
        # Because U_S might contain elements of S (it doesn't, by definition of tournament, but let's be safe).
        # Actually, u is not in out_neighbors[u], so U_S does not contain any u in S.
        # So U_S is disjoint from S.
        # So we need U_S == (all_mask ^ mask_S).
        
        for S in itertools.combinations(range(p), 4):
            mask_S = 0
            U_S = 0
            for u in S:
                mask_S |= (1 << u)
                U_S |= bitsets[u]
            
            # Check if U_S covers V\S
            # V\S is all_mask with S bits cleared
            target = all_mask ^ mask_S
            if (U_S & target) == target:
                # S dominates V\S, so this tournament is INVALID
                valid = False
                break
        
        if valid:
            # This Paley tournament is valid
            # Construct orientation list
            orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
            return {'n': p, 'orientation': orientation}
            
    # If no valid Paley prime found, return the fallback (83)
    return {'n': best_n, 'orientation': best_orientation}
