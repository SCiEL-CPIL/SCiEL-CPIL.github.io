def construct():
    """
    Find the minimum s (1 <= s <= 1004) such that s, s+1, s+2, s+3, s+4
    are all non-zero quadratic residues modulo p=1009.
    Return {'start': s, 'roots': [r0, r1, r2, r3, r4]} where ri**2 % 1009 == s+i.
    """
    p = 1009
    m = 5  # run length

    # Precompute quadratic residues and the smallest representative root for each.
    # For each x in 1..p-1, x*x % p is a QR. We store the first (smallest) root.
    root_map = {}  # residue -> smallest root r in 1..p-1
    is_qr = [False] * p  # is_qr[v] is True if v is a non-zero QR mod p

    for x in range(1, p):
        res = (x * x) % p
        if res != 0:
            if res not in root_map:
                root_map[res] = x
            is_qr[res] = True

    # s must satisfy 1 <= s <= p - m (i.e., s <= 1004) so that s+m-1 <= p-1
    # and all values s..s+m-1 are in 1..p-1 (non-zero, no wraparound).
    # Iterate s from 1 upward; the first valid s is the minimum.
    
    for s in range(1, p - m + 1):
        # Check if s, s+1, ..., s+m-1 are all QRs
        if all(is_qr[s + i] for i in range(m)):
            roots = [root_map[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    # Fallback: should not happen for p=1009
    raise RuntimeError("No valid run of 5 consecutive non-zero QRs found modulo 1009")
