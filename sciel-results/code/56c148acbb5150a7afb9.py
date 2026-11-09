"""
Solution for the task: Find the smallest start 's' such that s, s+1, s+2, s+3, s+4
are all non-zero quadratic residues modulo 1009.
Return the start s and the corresponding roots.
"""

def construct():
    P = 1009
    M = 5
    
    # Build a map from residue to one root
    # roots[r] = x such that x*x % P == r
    roots = {}
    qr_bits = 0
    for x in range(1, P):
        r = (x * x) % P
        # r is in 1..P-1
        if r not in roots:
            roots[r] = x
        # Set the bit corresponding to residue r
        qr_bits |= (1 << r)
        
    # Mask for M consecutive bits (5 bits)
    mask = (1 << M) - 1
    
    # Search for the minimum s
    # s must be in 1..1004 to keep s+4 <= 1008
    # This also ensures no wraparound to 0 or P
    for s in range(1, P - M + 1):
        # Check if bits s, s+1, ..., s+4 are all set in qr_bits
        # (qr_bits >> s) & mask extracts the bits at positions s..s+M-1
        if ((qr_bits >> s) & mask) == mask:
            current_roots = [roots[s + i] for i in range(M)]
            return {'start': s, 'roots': current_roots}
            
    # Fallback (should not happen)
    raise RuntimeError("No solution found")
