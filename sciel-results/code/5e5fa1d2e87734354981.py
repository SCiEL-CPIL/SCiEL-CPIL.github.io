"""
Rewritten program: find the minimal start s for five consecutive quadratic residues mod 1009.
Optimized for speed and clarity.
"""

def construct():
    p = 1009
    
    # Precompute quadratic residues and their smallest positive roots
    # is_qr[r] is True if r is a non-zero quadratic residue mod p
    # root_map[r] stores a root x such that x*x % p == r
    is_qr = [False] * p
    root_map = {}
    
    for x in range(1, p):
        r = (x * x) % p
        is_qr[r] = True
        if r not in root_map:
            root_map[r] = x
            
    # Search for the smallest s in 1..1004
    # s+i must be in 1..1008, so s <= 1004
    for s in range(1, p - 4):
        # Check if s, s+1, s+2, s+3, s+4 are all non-zero QRs
        # Since s starts at 1, s+i is always >= 1.
        # Since s <= 1004, s+4 <= 1008 < p, so no wraparound to 0 or negative.
        if (is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]):
            roots = [
                root_map[s],
                root_map[s+1],
                root_map[s+2],
                root_map[s+3],
                root_map[s+4]
            ]
            return {'start': s, 'roots': roots}
            
    # This should never happen for p=1009 as s=1 is a valid solution
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
