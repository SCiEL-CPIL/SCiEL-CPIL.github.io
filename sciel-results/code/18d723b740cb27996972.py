import time

P = 1009
M = 5

def construct():
    """
    Finds the minimal start s such that s, s+1, ..., s+4 are all non-zero 
    quadratic residues modulo 1009.
    Returns a dictionary with 'start' and 'roots'.
    """
    t_start = time.time()
    # The problem is extremely fast, but we adhere to the compute budget directive.
    deadline = t_start + 85.0 

    # roots[r] = x such that x^2 % P == r, with 1 <= x <= P-1
    # Initialize with 0, which is not a valid root (roots must be 1..1008)
    roots = [0] * P
    
    # Precompute roots for all non-zero quadratic residues
    # We iterate x from 1 to P-1. 
    # Note: x and P-x give the same residue. We just need one root per residue.
    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            if roots[r] == 0:
                roots[r] = x

    # Build bitmask: bit v is set if v is a non-zero quadratic residue
    # Bit 0 is for residue 0, which is excluded.
    mask = 0
    for r in range(1, P):
        if roots[r] != 0:
            mask |= (1 << r)

    # Target mask for M=5 consecutive bits: 0b11111 = 31
    target = (1 << M) - 1

    # Search for minimal s in [1, P-M]
    # s must be <= 1004 so that s+4 <= 1008, avoiding wraparound and zero residue issues
    # (Residue 0 is at index 0, and we exclude it. Max index is 1008).
    for s in range(1, P - M + 1):
        # Check if bits s, s+1, s+2, s+3, s+4 are all set in mask
        if (mask >> s) & target == target:
            # Time guard check (should never trigger for this small P)
            if time.time() > deadline:
                break
            
            result_roots = [roots[s + i] for i in range(M)]
            return {'start': s, 'roots': result_roots}

    # Fallback: If the loop broke due to time or didn't find anything (should not happen)
    # Re-scan without time check just in case, though this is redundant for P=1009
    for s in range(1, P - M + 1):
        if (mask >> s) & target == target:
            result_roots = [roots[s + i] for i in range(M)]
            return {'start': s, 'roots': result_roots}

    # Ultimate fallback (mathematically impossible for this P and M)
    # Return a valid structure to avoid exceptions, though values will be invalid
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}
