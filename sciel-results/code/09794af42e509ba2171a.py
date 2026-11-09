"""Find the minimal start s for 5 consecutive quadratic residues modulo 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    p = 1009
    m = 5

    # Build mapping: residue -> one root (1..p-1) for each non-zero QR
    roots = {}
    mask = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if r not in roots:
                roots[r] = x
            mask |= (1 << r)

    # Create a mask for a window of m consecutive bits
    window_mask = (1 << m) - 1

    # Search for minimum s in 1..p-m such that s..s+m-1 are all QRs
    # We shift the mask right by s and check if the lower m bits are all 1
    for s in range(1, p - m + 1):
        if ((mask >> s) & window_mask) == window_mask:
            result_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': result_roots}

    raise RuntimeError('No run of 5 consecutive QRs found')

# EVOLVE-BLOCK-END
