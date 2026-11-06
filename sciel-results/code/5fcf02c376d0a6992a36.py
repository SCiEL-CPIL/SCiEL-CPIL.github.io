"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute one root for each non-zero quadratic residue modulo p
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            roots[r] = x
    
    # Build a boolean array: is_res[i] = True if i is a non-zero QR mod p
    is_res = [False] * (p + 1)  # indices 0..p
    for r in roots:
        is_res[r] = True
    
    # Scan s from 1 upward to find the minimal valid start
    for s in range(1, p - 4):  # s+4 <= 1008, so s <= 1004
        if (is_res[s] and is_res[s+1] and is_res[s+2] and is_res[s+3] and is_res[s+4]):
            return {'start': s, 'roots': [roots[s+i] for i in range(5)]}
    
    raise RuntimeError('No run found')
# EVOLVE-BLOCK-END
