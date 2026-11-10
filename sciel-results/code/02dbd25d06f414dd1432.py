"""Find minimum s such that s, s+1, s+2, s+3, s+4 are all
non-zero quadratic residues mod 1009, each certified by a root in [1,1008]."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute: for each non-zero residue r in [1, p-1], store a root x in [1, p-1]
    # such that x^2 % p == r. Keep the smallest x for determinism.
    root_of = {}
    for x in range(1, p):
        if time.time() > deadline:
            break
        r = (x * x) % p
        if r != 0 and r not in root_of:
            root_of[r] = x

    # Build a boolean membership array for O(1) array-index lookups instead of dict lookups.
    # is_res[r] is True iff r is a non-zero quadratic residue mod p.
    # Extra slots to safely index up to p+4 without bounds checks.
    is_res = [False] * (p + m + 1)
    for r in root_of:
        is_res[r] = True

    # Scan for the minimum s in [1, p-m] such that s, s+1, ..., s+m-1 are all non-zero QRs.
    # Since p=1009 and m=5, s ranges from 1 to 1004, and s+4 <= 1008, so no wraparound.
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
        # Fast membership check via boolean array
        if (is_res[s] and is_res[s + 1] and is_res[s + 2] and is_res[s + 3] and is_res[s + 4]):
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    # Fallback: should not happen, but return a valid late run if we must.
    for s in range(p - m, 0, -1):
        if time.time() > deadline:
            break
        ok = True
        for i in range(m):
            val = s + i
            if val not in root_of:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [root_of[s + i] for i in range(m)]}

    raise RuntimeError('No run found')
# EVOLVE-BLOCK-END
