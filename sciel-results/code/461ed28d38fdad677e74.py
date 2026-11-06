"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Compute all quadratic residues mod p and store one root for each
    # Using the fact that for prime p, (a/p) = a^((p-1)/2) mod p is 1 if a is a QR
    # But we can also just compute x^2 mod p for x in 1..(p-1)//2 and collect
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Store the smallest root (or any root)
            if r not in roots:
                roots[r] = x
    
    # Now scan from s=1 upwards to find the first valid run of 5 consecutive QRs
    # s, s+1, s+2, s+3, s+4 must all be non-zero quadratic residues
    # Note: s can be at most 1004 (since s+4 <= 1008, and residues are 1..1008)
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all in roots
        if (s in roots and 
            (s+1) in roots and 
            (s+2) in roots and 
            (s+3) in roots and 
            (s+4) in roots):
            return {'start': s, 'roots': [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
