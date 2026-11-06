import time

def construct():
    """
    Find the minimum start s (1 <= s <= 1004) such that s, s+1, ..., s+4 are
    all quadratic residues modulo 1009, with explicit non-zero roots for each.
    
    Strategy:
    1. Precompute a boolean array is_qr where is_qr[r] is True if r is a non-zero quadratic residue mod 1009.
    2. Precompute a roots dict mapping each non-zero QR to its smallest positive root.
    3. Scan s from 1 upwards. For each s, check if all of s..s+4 are in the residue set.
    4. Return the first valid s with its roots.
    
    This is O(p) preprocessing and O(p) scanning, which is negligible.
    """
    p = 1009
    deadline = time.time() + 50  # Safety margin well below 60s limit

    # Precompute roots: for each residue r, store one root x (1 <= x <= 1008)
    # such that x^2 mod p == r. We want the smallest such x for determinism.
    roots = {}
    is_qr = [False] * p  # is_qr[r] = True if r is a non-zero QR mod p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:  # Exclude zero residues
            is_qr[r] = True
            if r not in roots:  # Keep the first (smallest) root found
                roots[r] = x

    # Scan for the minimum start s
    # s ranges from 1 to 1004 (so that s+4 <= 1008, no wraparound into 1009+)
    for s in range(1, 1005):
        if time.time() > deadline:
            # Should never happen for this small p, but guard against infinite loops or stalls
            break
        
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        # Using list comprehension for clarity and slight speedup over nested loops
        if all(is_qr[s + i] for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
            
    raise RuntimeError('No valid run found')
