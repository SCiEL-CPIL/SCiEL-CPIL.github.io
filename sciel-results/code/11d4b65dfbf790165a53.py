def construct():
    p = 1009
    # Precompute quadratic residues and a representative root for each
    # We only care about residues 1..1008 (non-zero)
    qr_set = set()
    root_map = {}
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            qr_set.add(r)
            # Store the first (smallest) root found for determinism
            if r not in root_map:
                root_map[r] = x
    
    # Search for the smallest s such that s, s+1, s+2, s+3, s+4 are all QRs
    # s must be in 1..1004
    for s in range(1, p - 4):
        valid = True
        for i in range(5):
            val = s + i
            if val not in qr_set:
                valid = False
                break
        
        if valid:
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
    
    # Fallback (should not happen for this prime and m=5)
    raise RuntimeError("No valid run found")
