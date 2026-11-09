"""Find the minimal start s for 5 consecutive quadratic residues modulo 1009."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    p = 1009
    m = 5

    # Build mapping: residue -> one root (1..p-1) for each non-zero QR
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            roots.setdefault(r, x)

    # Search for minimum s in 1..p-m such that s..s+m-1 are all QRs
    for s in range(1, p - m + 1):
        if all((s + i) in roots for i in range(m)):
            result_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': result_roots}

    raise RuntimeError('No run of 5 consecutive QRs found')

# EVOLVE-BLOCK-END
