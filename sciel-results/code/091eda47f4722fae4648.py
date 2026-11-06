"""Find the minimal start s for 5 consecutive quadratic residues mod 1009."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Build a bitset where bit i is set if i is a non-zero quadratic residue mod p
    # Also store one root for each residue
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x  # store the first (smallest) root found
    
    # Build bitset as an integer: bit i set means i is a QR (non-zero)
    # We need bits 1..1008 (indices 0..1008 in the integer)
    bitset = 0
    for r in roots:
        bitset |= (1 << r)
    
    # We need to find smallest s in [1, 1004] such that bits s, s+1, s+2, s+3, s+4 are all set
    # Check: (bitset >> s) & 0b11111 == 0b11111
    mask = 0b11111  # 5 bits set
    
    for s in range(1, 1005):
        if (bitset >> s) & mask == mask:
            # All five residues s..s+4 are non-zero QRs
            result_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': result_roots}
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
# EVOLVE-BLOCK-END
