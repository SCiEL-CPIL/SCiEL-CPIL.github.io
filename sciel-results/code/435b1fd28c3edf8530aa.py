import time

def construct():
    p = 1009
    deadline = time.time() + 59  # 59 seconds safety margin
    
    # 1. Precompute quadratic residues and their smallest roots
    # roots[r] = smallest x in [1, p-1] such that x*x % p == r
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if r not in roots:
                roots[r] = x
        
    # Check time
    if time.time() > deadline:
        # Fallback to a known valid solution if we ran out of time (unlikely)
        # But we are just building a dict, this is very fast.
        pass

    # 2. Build a bitset where bit i is 1 if i is a non-zero quadratic residue
    bitset = 0
    for r in roots:
        bitset |= (1 << r)
        
    # 3. Find the minimum s such that s, s+1, s+2, s+3, s+4 are all residues
    # This is equivalent to finding the smallest set bit in:
    # mask = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    
    mask = bitset
    mask &= (bitset >> 1)
    mask &= (bitset >> 2)
    mask &= (bitset >> 3)
    mask &= (bitset >> 4)
    
    # s must be >= 1, so clear bit 0
    mask &= ~1
    
    if mask == 0:
        # Fallback: linear scan (should not happen for p=1009)
        for s in range(1, p - 4):
            if all((s + i) in roots for i in range(5)):
                break
        else:
            raise RuntimeError('No valid run found')
    else:
        # Find the least significant set bit
        # mask & -mask isolates the lowest set bit
        lowest_bit = mask & -mask
        s = lowest_bit.bit_length() - 1
        
    # 4. Construct the result
    result_roots = [roots[s + i] for i in range(5)]
    
    return {'start': s, 'roots': result_roots}
