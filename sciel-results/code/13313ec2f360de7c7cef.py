def construct():
    p = 1009
    roots = {}
    mask = 0
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
        mask |= (1 << r)
    # Find the lowest bit position s such that bits s..s+4 are all set in mask
    consec = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    # Get the lowest set bit position
    lsb = consec & (-consec)
    s = lsb.bit_length() - 1
    # Build the result
    result_roots = []
    for i in range(5):
        result_roots.append(roots[s + i])
    return {'start': s, 'roots': result_roots}
