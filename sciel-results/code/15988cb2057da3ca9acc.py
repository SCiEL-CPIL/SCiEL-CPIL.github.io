"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    p = 1009
    # Precompute quadratic residues and their roots
    # We want the smallest s, so we just need any valid root for each residue.
    # The problem doesn't specify minimizing the roots, just finding valid ones.
    # "Each residue is certified by its root".
    # We'll store one root for each residue.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
    
    # The key insight: we need to find the smallest s such that s, s+1, s+2, s+3, s+4
    # are all non-zero quadratic residues modulo 1009.
    # We can use a bitmask approach for efficiency.
    # Create a bitmask where bit v is set if v is a non-zero quadratic residue.
    mask = 0
    for r in roots:
        if r != 0:
            mask |= (1 << r)
    
    # Now iterate s from 1 to 1004, checking if bits s through s+4 are all set.
    # (mask >> s) & 0x1F == 0x1F means all 5 consecutive residues exist.
    for s in range(1, 1005):
        if time.time() > deadline:
            # Fallback: return the first valid s we can find quickly
            # This shouldn't happen for p=1009, but just in case
            for s2 in range(1, 1005):
                if all((s2+i) in roots and (s2+i) != 0 for i in range(5)):
                    return {'start': s2, 'roots': [roots[s2+i] for i in range(5)]}
            raise RuntimeError('No baseline run found')
        
        if ((mask >> s) & 0x1F) == 0x1F:
            return {'start': s, 'roots': [roots[s+i] for i in range(5)]}
    
    raise RuntimeError('No baseline run found')
# EVOLVE-BLOCK-END
