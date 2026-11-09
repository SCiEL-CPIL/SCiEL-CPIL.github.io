def construct():
    p = 1009
    # Precompute quadratic residues and one root for each
    is_residue = [False] * p
    roots = {}
    
    for x in range(1, p):
        v = (x * x) % p
        is_residue[v] = True
        # Store the first root found. Since we iterate x from 1, 
        # this will be the smaller of the two roots (x, p-x) if x < p/2,
        # or just a valid root. It doesn't matter which valid root we pick.
        if v not in roots:
            roots[v] = x
            
    # Search for the smallest s such that s, s+1, s+2, s+3, s+4 are all residues
    # s must be at least 1.
    # s+4 must be at most p-1 (1008) to avoid wraparound issues if we were checking modulo,
    # but here we are checking integer values in 1..1008.
    # The problem states 1 <= s <= 1004, so s+4 <= 1008.
    
    for s in range(1, p - 4):
        if all(is_residue[s + i] for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
            
    # This should never happen for p=1009 as there are many runs of residues
    raise RuntimeError("No run of 5 consecutive quadratic residues found")
