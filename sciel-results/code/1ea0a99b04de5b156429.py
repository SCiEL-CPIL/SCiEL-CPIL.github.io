import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest s in 1..1004 such that s, s+1, s+2, s+3, s+4 are all
    quadratic residues modulo 1009 (non-zero), and return the start and roots.
    """
    p = 1009
    m = 5
    
    # Precompute the smallest root for each quadratic residue in 1..p-1
    # For each x in 1..p-1, x^2 mod p is a residue. We want the smallest root.
    # Note: x and p-x give the same residue. So we only need x in 1..p//2.
    root_map = {}
    for x in range(1, p // 2 + 1):
        r = (x * x) % p
        if r != 0:
            if r not in root_map:
                root_map[r] = x
            # x is smaller than p-x, and we're going up, so first one is smallest
    
    # Now search for the smallest s such that s, s+1, ..., s+4 are all in root_map
    # and s+i != 0 for all i (which is guaranteed since s>=1 and s+4<=1008)
    # Also s+i must be in 1..1008
    
    for s in range(1, p - m + 1):  # s from 1 to 1004
        valid = True
        for i in range(m):
            val = s + i
            if val not in root_map:
                valid = False
                break
        if valid:
            roots = [root_map[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
