"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    
    p = 1009
    n = 1008  # residues 1..1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x where x^2 ≡ 1 mod p => x=1 or x=p-1=1008
    # These cannot be in any valid interval
    
    # For x in [2, 1007], inv[x] is also in [2, 1007] and inv[x] != x
    # We need the longest subinterval of [2, 1007] that doesn't contain both x and inv[x] for any x.
    
    # For a fixed start a, max valid b = min(1007, min{inv[x] - 1 : x in [a, 1007], inv[x] >= a})
    # If no such x exists, max_b(a) = 1007
    
    best_len = 0
    best_start = 2
    
    # For efficiency, we can precompute for each position x, the value inv[x]
    # Then for each a, we need to find the minimum inv[x]-1 over x in [a, 1007] where inv[x] >= a
    
    # Let's use a segment tree or sparse table for range minimum queries
    # But since n is only 1008, O(n^2) is fine (~500K operations)
    
    # Alternative: for each a, scan from a to 1007 and track the minimum constraint
    # We can break early if the constraint becomes < a (impossible)
    
    for a in range(2, 1008):
        if time.time() > deadline:
            break
        
        # Find the minimum inv[x]-1 for x in [a, 1007] where inv[x] >= a
        min_constraint = 1007  # default: no constraint beyond 1007
        for x in range(a, 1008):
            ix = inv[x]
            if ix >= a:
                c = ix - 1
                if c < min_constraint:
                    min_constraint = c
                    if min_constraint < a:
                        break  # can't go below a
        
        b = min(1007, min_constraint)
        if b >= a:
            length = b - a + 1
            if length > best_len:
                best_len = length
                best_start = a
    
    # Verify the result is valid
    start = best_start
    length = best_len
    end = start + length - 1
    
    # Double-check: for all x in [start, end], inv[x] should not be in [start, end]
    valid = True
    for x in range(start, end + 1):
        ix = inv[x]
        if start <= ix <= end:
            valid = False
            break
        if x * x % p == 1:
            valid = False
            break
    
    if not valid:
        # Fallback: return a known valid small interval
        # [2, 2] is valid since inv[2] = 505, which is not in [2,2]
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
