"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Build a bitset of quadratic residues modulo p (excluding 0).
    # Also store one root for each residue in a list for O(1) lookup.
    
    residues = 0
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residues |= (1 << r)
            if roots[r] == 0:
                roots[r] = x
    
    # Compute bitset "runs" where bit s is set if s, s+1, ..., s+4 are all QRs.
    runs = residues
    for i in range(1, 5):
        runs &= (residues >> i)
    
    # Find the smallest s >= 1 such that bit s is set in runs.
    # Mask out bit 0 to ensure s >= 1.
    m = runs & ~1
    
    if m == 0:
        raise RuntimeError('No run of 5 consecutive quadratic residues found')
    
    # Find index of least significant set bit in m
    s = (m & -m).bit_length() - 1
    
    # Verify s is within valid range [1, 1004]
    if s > 1004:
        raise RuntimeError('Start index out of bounds')
        
    root_list = [roots[s + i] for i in range(5)]
    return {'start': s, 'roots': root_list}
# EVOLVE-BLOCK-END
