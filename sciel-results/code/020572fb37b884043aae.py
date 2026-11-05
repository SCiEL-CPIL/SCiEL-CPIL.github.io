"""Rewritten program: find the minimal start s for five consecutive quadratic residues mod 1009 using bitmask."""

def construct():
    p = 1009
    # Precompute roots
    root_map = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in root_map:
            root_map[r] = x
            
    # Build bitmask for non-zero quadratic residues
    # Bit i is set if i is a QR
    mask = 0
    for r in root_map.keys():
        mask |= (1 << r)
        
    # Search for s
    # We need bits s, s+1, s+2, s+3, s+4 to be set.
    # This is equivalent to ((mask >> s) & 0b11111) == 0b11111
    for s in range(1, p - 4):
        if ((mask >> s) & 0x1F) == 0x1F:
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
