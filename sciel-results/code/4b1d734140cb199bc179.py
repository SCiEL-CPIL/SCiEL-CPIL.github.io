import time

def construct():
    p = 1009
    m = 5
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin

    # Precompute roots for each non-zero quadratic residue
    # We want the smallest positive root for each residue
    # Use a list for O(1) lookup and initialization
    roots_list = [0] * p
    
    # Build bitset for quadratic residues
    # Bit i is 1 if i is a non-zero quadratic residue
    res = 0
    for x in range(1, p):
        r = (x * x) % p
        if roots_list[r] == 0:
            roots_list[r] = x
        res |= (1 << r)
        
    # We need s, s+1, s+2, s+3, s+4 to all be residues.
    # This is equivalent to finding the lowest set bit in:
    # res & (res >> 1) & (res >> 2) & (res >> 3) & (res >> 4)
    
    mask = res & (res >> 1) & (res >> 2) & (res >> 3) & (res >> 4)
    
    if mask == 0:
        # Fallback to linear search if bitset logic fails (should not happen)
        # This path is unlikely to be taken given the math
        for s in range(1, p - m + 1):
            valid = True
            for i in range(m):
                if roots_list[s + i] == 0:
                    valid = False
                    break
            if valid:
                return {'start': s, 'roots': [roots_list[s+i] for i in range(m)]}
        raise RuntimeError("No run found")
        
    # Find the index of the lowest set bit
    # mask & -mask isolates the lowest set bit
    lowest_bit = mask & -mask
    s = lowest_bit.bit_length() - 1
    
    # Verify s is within bounds
    # The bitset construction ensures s+4 <= 1008 because bits > 1008 are 0
    # and s >= 1 because bit 0 of res is 0.
    
    res_roots = [roots_list[s + i] for i in range(m)]
    
    return {'start': s, 'roots': res_roots}
