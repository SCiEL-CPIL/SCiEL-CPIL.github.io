def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85
    
    P = 1009
    M = 5
    
    # Precompute quadratic residues modulo P
    # qr_root[x] = r such that r^2 % P == x, for x in 1..P-1
    qr_root = [0] * P
    for r in range(1, P):
        x = (r * r) % P
        if x != 0 and qr_root[x] == 0:
            qr_root[x] = r
    
    # Build a bitmask where bit i is set if i is a quadratic residue
    mask = 0
    for x in range(1, P):
        if qr_root[x] != 0:
            mask |= (1 << x)
            
    # We need to find s such that s, s+1, ..., s+4 are all QRs
    # This corresponds to bits s, s+1, s+2, s+3, s+4 being set in mask
    # We can check this by shifting mask right by s and checking the lowest 5 bits
    # (mask >> s) & 0x1F == 0x1F
    
    # Search for the smallest s in [1, P-M]
    for s in range(1, P - M + 1):
        if time.time() > deadline:
            break
        # Check if 5 consecutive bits are set starting at s
        if ((mask >> s) & 0x1F) == 0x1F:
            roots = [qr_root[s + i] for i in range(M)]
            return {'start': s, 'roots': roots}
            
    # Fallback
    return {'start': 1, 'roots': [1, 1, 1, 1, 1]}
