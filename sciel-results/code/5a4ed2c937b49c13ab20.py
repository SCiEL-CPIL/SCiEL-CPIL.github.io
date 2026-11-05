P = 1009
M = 5

def construct():
    """
    Finds the minimal start s such that s, s+1, ..., s+4 are all non-zero 
    quadratic residues modulo 1009.
    Returns a dictionary with 'start' and 'roots'.
    """
    # roots[r] = x such that x^2 % P == r, with 1 <= x <= P-1
    # Initialize with 0, which is not a valid root (roots must be 1..1008)
    roots = [0] * P
    
    # Precompute roots for all non-zero quadratic residues
    for x in range(1, P):
        r = (x * x) % P
        if r != 0 and roots[r] == 0:
            roots[r] = x

    # Build bitmask: bit v is set if v is a non-zero quadratic residue
    mask = 0
    for r in range(1, P):
        if roots[r] != 0:
            mask |= (1 << r)

    # Target mask for M=5 consecutive bits: 0b11111 = 31
    target = (1 << M) - 1

    # Search for minimal s in [1, P-M]
    for s in range(1, P - M + 1):
        if (mask >> s) & target == target:
            result_roots = [roots[s + i] for i in range(M)]
            return {'start': s, 'roots': result_roots}

    # Fallback (should not happen)
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}
