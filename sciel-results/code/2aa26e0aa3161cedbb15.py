def construct():
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
    
    # The set of quadratic residues
    # Note: 0 is a quadratic residue (0^2), but the problem excludes zero residues.
    # Our range for s+i is 1..1008, so 0 is not in the values s+i.
    # So we just check membership in the keys of 'roots'.
    qr_set = set(roots.keys())
    
    # Search for the smallest s
    # s must be such that s, s+1, ..., s+4 are all in qr_set
    # s is in [1, 1004]
    for s in range(1, p - 4):
        if all((s + i) in qr_set for i in range(5)):
            # Found the minimal s
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    # If no such s is found (should not happen for p=1009)
    raise RuntimeError("No run of 5 consecutive quadratic residues found")
