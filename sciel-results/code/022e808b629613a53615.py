def construct():
    p = 1009
    # roots[i] will store the smallest x in [1, p-1] such that x^2 % p == i
    # Initialize with 0
    roots = [0] * p
    
    # Build residue mask
    # Bit k is 1 if k is a quadratic residue
    mask = 0
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Set bit r in mask
            mask |= (1 << r)
            # Store smallest root
            if roots[r] == 0 or x < roots[r]:
                roots[r] = x

    # Find s
    # We need s, s+1, s+2, s+3, s+4 to be residues
    # This means bit s, s+1, s+2, s+3, s+4 in mask are all 1
    # common = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    common = mask
    common &= mask >> 1
    common &= mask >> 2
    common &= mask >> 3
    common &= mask >> 4
    
    # Find the least significant set bit in common
    # If common is 0, no solution found (should not happen)
    if common == 0:
        raise RuntimeError("No run of 5 quadratic residues found")
    
    # Isolate lowest set bit
    low_bit = common & -common
    s = low_bit.bit_length() - 1
    
    # Verify s is in range [1, 1004]
    # The construction of mask ensures bits 0 and >= 1009 are 0.
    # So s will be in [1, 1004] if a solution exists.
    
    res_roots = [roots[s + i] for i in range(5)]
    
    return {'start': s, 'roots': res_roots}
