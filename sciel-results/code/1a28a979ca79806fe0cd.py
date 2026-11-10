def construct():
    p = 1009
    # Precompute smallest root for each non-zero quadratic residue
    root_map = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in root_map:
            root_map[r] = x
            
    # Set of quadratic residues for fast membership testing
    qr_set = set(root_map.keys())
    
    # Search for minimum s
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all in qr_set
        if all((s + i) in qr_set for i in range(5)):
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError('No valid run found')
