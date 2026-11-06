import time
from itertools import combinations

def construct():
    """
    Returns a list of 101 lists. Each list contains distinct denominators in 1..10
    such that the sum of their modular inverses modulo 101 equals the index.
    Minimizes the largest row length.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit
    
    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inv = [0] * (MAX_DENOM + 1)
    for d in denominators:
        inv[d] = pow(d, -1, MOD)
    
    # For each residue, find the shortest subset that sums to it
    best = [None] * MOD  # best[r] = (length, list_of_denoms)
    
    # Iterate by length to find shortest first
    for length in range(1, MAX_DENOM + 1):
        for combo in combinations(denominators, length):
            s = 0
            for d in combo:
                s = (s + inv[d]) % MOD
            r = s
            if best[r] is None:
                best[r] = (length, list(combo))
        
        # Check if all residues are covered
        if all(b is not None for b in best):
            break
        
        # Deadline guard
        if time.time() - start_time > DEADLINE:
            break
    
    # Build results
    results = []
    for r in range(MOD):
        if best[r] is not None:
            results.append(best[r][1])
        else:
            # Fallback: find any subset
            found = False
            for length in range(1, MAX_DENOM + 1):
                for combo in combinations(denominators, length):
                    s = 0
                    for d in combo:
                        s = (s + inv[d]) % MOD
                    if s == r:
                        results.append(list(combo))
                        found = True
                        break
                if found:
                    break
            if not found:
                results.append([1])
    
    return results
