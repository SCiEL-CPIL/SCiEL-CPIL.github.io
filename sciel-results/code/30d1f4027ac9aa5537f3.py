"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _paley_tournament(p):
    """
    Returns orientation list for Paley tournament on p vertices.
    p must be a prime congruent to 3 mod 4.
    Orientation bit for pair (i, j) with i < j:
    0 means i -> j
    1 means j -> i
    """
    # Precompute squares modulo p
    squares = set((x * x) % p for x in range(1, p))
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            d = (j - i) % p
            # In Paley tournament, i -> j if (j-i) is a quadratic residue mod p
            # Here 0 means i -> j, 1 means j -> i
            if d in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}

def _fast_check_property(n, orientation):
    """
    Check if every 4-set has a common dominator using bitmasks.
    Returns True if valid.
    """
    out_mask = [0] * n
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                out_mask[i] |= (1 << j)
            else:
                out_mask[j] |= (1 << i)
    
    # Optimization: Precompute list of valid 4-tuples? No, just iterate.
    # We can skip 4-sets where no vertex has out-degree >= 3 within the set.
    # But bitmask check is very fast.
    
    for a, b, c, d in itertools.combinations(range(n), 4):
        mask_abcd = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        
        if (out_mask[a] & (mask_abcd ^ (1 << a))) != (mask_abcd ^ (1 << a)):
            if (out_mask[b] & (mask_abcd ^ (1 << b))) != (mask_abcd ^ (1 << b)):
                if (out_mask[c] & (mask_abcd ^ (1 << c))) != (mask_abcd ^ (1 << c)):
                    if (out_mask[d] & (mask_abcd ^ (1 << d))) != (mask_abcd ^ (1 << d)):
                        return False
    return True

def construct():
    # The smallest known valid Paley tournament for Schütte's problem (k=4) is on p=67.
    # Paley tournaments on smaller primes (31, 43, 47, 59) do not satisfy the property.
    # Returning the known valid tournament immediately ensures correctness and minimizes runtime.
    return _paley_tournament(67)
# EVOLVE-BLOCK-END
