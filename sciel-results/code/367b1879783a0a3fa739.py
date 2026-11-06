"""Find the minimum start s for 5 consecutive quadratic residues mod 1009."""

def construct():
    p = 1009
    
    # Precompute all non-zero quadratic residues and their smallest positive root
    # For each residue r (1..1008), store the smallest x in 1..1008 such that x^2 % p == r
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if r not in roots or x < roots[r]:
            roots[r] = x
    
    # Search for the minimum s such that s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # and s+4 <= 1008 (i.e., s <= 1004)
    for s in range(1, 1005):
        res_roots = []
        valid = True
        for i in range(5):
            val = s + i
            if val in roots:
                res_roots.append(roots[val])
            else:
                valid = False
                break
        if valid:
            return {'start': s, 'roots': res_roots}
    
    # Fallback: should not happen for p=1009, but just in case
    raise RuntimeError('No valid run found')
