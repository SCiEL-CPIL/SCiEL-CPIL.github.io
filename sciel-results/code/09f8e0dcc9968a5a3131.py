"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    mask = 0
    # Build bitmask of non-zero quadratic residues
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            mask |= (1 << r)
            
    # Find smallest s such that s, s+1, s+2, s+3, s+4 are all residues
    s_found = None
    for s in range(1, 1005):
        if (mask >> s) & 0x1F == 0x1F:
            s_found = s
            break
            
    if s_found is None:
        raise RuntimeError('No baseline run found')
        
    # Find roots for the specific residues s_found .. s_found+4
    targets = {s_found + i for i in range(5)}
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r in targets:
            if r not in roots:
                roots[r] = x
                if len(roots) == 5:
                    break
                    
    return {'start': s_found, 'roots': [roots[s_found + i] for i in range(5)]}
# EVOLVE-BLOCK-END
