import time

def construct():
    deadline = time.time() + 85.0
    p = 1009
    
    # Build the bitset of non-zero quadratic residues
    # bit i is 1 if i is a QR mod p and i != 0
    res_mask = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            res_mask |= (1 << r)
            
    # We need s, s+1, s+2, s+3, s+4 to be QRs.
    # This means bit s, s+1, s+2, s+3, s+4 in res_mask must be 1.
    # M = res_mask & (res_mask >> 1) & (res_mask >> 2) & (res_mask >> 3) & (res_mask >> 4)
    
    M = res_mask
    for i in range(1, 5):
        M &= (res_mask >> i)
        
    # Find the smallest s >= 1 such that bit s is set in M.
    # M has bit s set iff s..s+4 are QRs.
    # We want the lowest set bit.
    
    if M == 0:
        # No solution found in the bitset logic?
        # This shouldn't happen for p=1009.
        # Fallback to linear scan just in case.
        s = -1
        for candidate_s in range(1, p - 4):
            if all(((res_mask >> (candidate_s + i)) & 1) for i in range(5)):
                s = candidate_s
                break
        if s == -1:
            raise RuntimeError("No solution found")
    else:
        lowest_bit = M & -M
        s = lowest_bit.bit_length() - 1
        
        # Ensure s is within bounds
        if s < 1:
            # If bit 0 was set, it means 0..4 were QRs.
            # But 0 is not in res_mask, so bit 0 is 0.
            # So s should be >= 1.
            # If s is somehow 0, we need to find the next one.
            # Mask out bits < 1?
            # M & ~(1 << 0) ? No, we want the lowest set bit >= 1.
            # If bit 0 is 0, lowest_bit is >= 2 (bit 1).
            pass
            
        if s > 1004:
            # If s > 1004, we failed.
            pass

    # Find roots for s..s+4
    roots = []
    # Precompute roots
    root_of_res = [0] * (p + 1)
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_of_res[r] == 0:
            root_of_res[r] = x
            
    for i in range(5):
        val = s + i
        r = root_of_res[val]
        if r == 0:
            raise RuntimeError(f"No root for {val}")
        roots.append(r)
        
    return {'start': s, 'roots': roots}
