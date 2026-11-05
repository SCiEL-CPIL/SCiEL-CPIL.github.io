"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Use a list of size p+1 for O(1) root lookup instead of dict
    # roots[r] = smallest x in 1..p-1 such that x*x % p == r, or 0 if not a residue
    roots = [0] * (p + 1)
    mask = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            mask |= (1 << r)
            if roots[r] == 0:
                roots[r] = x
    
    # Search for smallest s in 1..1004 where bits s..s+4 are all set
    for s in range(1, 1005):
        if (mask >> s) & 0x1F == 0x1F:
            return {'start': s, 'roots': [roots[s+i] for i in range(5)]}
    
    raise RuntimeError('No baseline run found')
# EVOLVE-BLOCK-END
