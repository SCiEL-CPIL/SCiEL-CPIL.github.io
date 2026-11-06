# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Use a boolean array to mark quadratic residues
    is_qr = [False] * p
    root_for_residue = [0] * p  # Store one root for each residue
    
    for x in range(1, p):
        r = (x * x) % p
        is_qr[r] = True
        if root_for_residue[r] == 0:
            root_for_residue[r] = x
            
    # Scan for the minimal s
    # s must be such that s, s+1, ..., s+4 are all QRs and non-zero
    # Since s >= 1 and s+4 <= 1008, we don't hit 0 or wrap around >1008
    for s in range(1, 1005):
        # Check if s..s+4 are all QRs
        # We can unroll or use all()
        if all(is_qr[s+i] for i in range(5)):
            roots = [root_for_residue[s+i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
