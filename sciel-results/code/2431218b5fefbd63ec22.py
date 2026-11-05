def construct():
    p = 1009
    # Build bitmask of non-zero quadratic residues
    # bit v is set iff v is a non-zero QR mod p
    mask = 0
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        mask |= (1 << r)
        if r not in roots:
            roots[r] = x

    # Search ascending from s=1 to s=1004 for the first run of 5 consecutive QRs
    # Check: bits s through s+4 are all set in mask
    # (mask >> s) & 0x1F == 0x1F  means bits s..s+4 are all 1
    for s in range(1, 1005):
        if s + 4 > 1008:
            break
        if (mask >> s) & 0x1F == 0x1F:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}

    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
