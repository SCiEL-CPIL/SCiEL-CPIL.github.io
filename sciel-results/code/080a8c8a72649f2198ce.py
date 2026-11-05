"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    mask = 0
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r:
            mask |= 1 << r
            if r not in roots:
                roots[r] = x
    for s in range(1, 1005):
        if (mask >> s) & 0x1F == 0x1F:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    raise RuntimeError('No baseline run found')
# EVOLVE-BLOCK-END
