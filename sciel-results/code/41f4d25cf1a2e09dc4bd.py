def construct():
    import time
    p = 1009
    m = 5
    deadline = time.monotonic() + 80.0  # 10s safety margin before 90s kill

    # Build bitmask of non-zero quadratic residues:
    # bit i is set iff i is a non-zero QR mod p
    mask = 0
    # Also build root lookup: for each residue, store one root in 1..p-1
    root = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        mask |= (1 << r)
        if root[r] == 0:
            root[r] = x

    # Precompute the 5-bit pattern: all five bits set
    pattern = (1 << m) - 1  # 0b11111 = 31

    # Slide a window of m bits over the mask, starting at bit position s
    # We need bits s, s+1, ..., s+m-1 all set
    # (mask >> s) & pattern == pattern means all m bits starting at s are 1
    #
    # We iterate s from 1 to p - m (inclusive), i.e., 1 to 1004.
    # This is the minimal-s search.
    for s in range(1, p - m + 1):
        if time.monotonic() > deadline:
            break
        if (mask >> s) & pattern == pattern:
            roots = [root[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
