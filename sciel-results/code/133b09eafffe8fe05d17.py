"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    m = 5
    
    # Precompute all quadratic residues and their smallest positive root
    # For each residue r (1..1008), store the smallest x in 1..1008 such that x^2 % p == r
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if r not in roots or x < roots[r]:
            roots[r] = r  # placeholder, will fix below
    
    # Actually, let's just store any valid root; we want the smallest x for each residue
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if r not in roots or x < roots[r]:
            roots[r] = x
    
    # Search for the minimum s such that s, s+1, ..., s+4 are all quadratic residues
    # and s+4 <= 1008 (i.e., s <= 1004)
    for s in range(1, 1005):
        res_roots = []
        valid = True
        for i in range(m):
            val = s + i
            if val > 1008:
                valid = False
                break
            if val in roots:
                res_roots.append(roots[val])
            else:
                valid = False
                break
        if valid:
            return {'start': s, 'roots': res_roots}
    
    # Fallback: should not happen for p=1009, but just in case
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
