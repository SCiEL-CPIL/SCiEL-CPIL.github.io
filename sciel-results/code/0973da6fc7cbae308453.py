import time

def construct():
    deadline = time.time() + 85.0
    
    p = 1009
    m = 5
    
    # Precompute roots for each non-zero quadratic residue
    # roots[r] will store a root x such that x^2 % p == r
    roots = [0] * p
    
    # Precompute the bitset mask for non-zero quadratic residues
    # Bit i is 1 if i is a non-zero QR mod p
    mask = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            roots[r] = x
            mask |= (1 << r)
    
    # We want to find the smallest s in 1..p-m such that s..s+m-1 are all QRs.
    # This is equivalent to checking if bits s, s+1, ..., s+m-1 in mask are all 1.
    # We can create a mask of m consecutive 1s: target_mask = (1 << m) - 1
    target_mask = (1 << m) - 1
    
    # Scan s from 1 to p-m
    for s in range(1, p - m + 1):
        # Check if the m bits starting at s are all set
        # (mask >> s) shifts the bit at position s to position 0
        # & target_mask checks if the first m bits are all 1
        if (mask >> s) & target_mask == target_mask:
            result_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': result_roots}
            
    # Fallback (should not happen)
    raise RuntimeError('No baseline run found')
