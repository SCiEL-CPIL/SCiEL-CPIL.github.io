"""Find the minimum s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    
    # Precompute quadratic residues and their roots
    # roots[v] = smallest x such that x^2 % p == v, or 0 if v is not a non-zero QR
    roots = [0] * p
    
    for x in range(1, p):
        r2 = (x * x) % p
        if roots[r2] == 0:
            roots[r2] = x
    
    # Build a bitset: bit i is 1 if i is a non-zero quadratic residue mod p
    mask = 0
    for v in range(1, p):
        if roots[v] != 0:
            mask |= (1 << v)
    
    # valid_mask: bit s is 1 if s, s+1, s+2, s+3, s+4 are all non-zero QRs
    # mask >> k has bit s set if mask had bit s+k set
    valid_mask = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    
    # Find the least significant set bit in valid_mask (this gives the minimum s)
    if valid_mask == 0:
        # Fallback: linear search
        for s in range(1, p - m + 1):
            if all(roots[s + i] != 0 for i in range(m)):
                return {'start': s, 'roots': [roots[s + i] for i in range(m)]}
        raise RuntimeError('No run of 5 consecutive non-zero QRs found')
    
    # Isolate the lowest set bit
    lsb = valid_mask & -valid_mask
    s = lsb.bit_length() - 1
    
    # Verify constraints: 1 <= s <= 1004
    if s < 1 or s > 1004:
        # Fallback linear search
        for s_check in range(1, p - m + 1):
            if all(roots[s_check + i] != 0 for i in range(m)):
                s = s_check
                break
        else:
            raise RuntimeError('No valid s found in range')
    
    result_roots = [roots[s + i] for i in range(m)]
    
    return {'start': s, 'roots': result_roots}
# EVOLVE-BLOCK-END
