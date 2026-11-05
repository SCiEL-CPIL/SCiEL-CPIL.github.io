"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 85.0  # Safety margin before 90s wall-clock limit
    
    # Precompute quadratic residues and store one valid root for each
    roots = [0] * (p + 1)  # roots[r] = x such that x*x % p == r, 0 if not a residue
    for x in range(1, p):
        r = (x * x) % p
        if roots[r] == 0:
            roots[r] = x
    
    # Build a bitmask of which residues (1..1008) are quadratic residues
    # bitmask bit i is set if residue i (1-indexed) is a QR
    # We need to check if s, s+1, s+2, s+3, s+4 are all QRs
    # Use a boolean array for fast lookup
    is_qr = [False] * (p + 1)
    for x in range(1, p):
        is_qr[(x * x) % p] = True
    
    # Scan s in ascending order to find the minimal valid start
    for s in range(1, 1005):
        if (is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]):
            r0 = roots[s]
            r1 = roots[s+1]
            r2 = roots[s+2]
            r3 = roots[s+3]
            r4 = roots[s+4]
            # Verify all roots are valid (1 <= ri <= 1008)
            if all(1 <= ri <= 1008 for ri in (r0, r1, r2, r3, r4)):
                return {'start': s, 'roots': [r0, r1, r2, r3, r4]}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
