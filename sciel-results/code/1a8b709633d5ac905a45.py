"""Rewritten program: find minimum s such that s, s+1, s+2, s+3, s+4 are all
non-zero quadratic residues mod 1009, each certified by a root in [1,1008]."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    m = 5
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute: for each non-zero residue r in [1, p-1], store a root x in [1, p-1]
    # such that x^2 % p == r.
    # We scan x from 1 to p-1; the first x that maps to a given residue is kept.
    # This is fast: p is only 1009.
    root_of = {}
    for x in range(1, p):
        if time.time() > deadline:
            break
        r = (x * x) % p
        if r != 0 and r not in root_of:
            root_of[r] = x

    # Now search for the minimum s in [1, 1004] such that
    # s, s+1, s+2, s+3, s+4 are all in root_of (i.e., non-zero QRs).
    # Since p=1009, s+4 <= 1008, so no wraparound needed.
    #
    # The search space is small (1004 candidates), and each check is O(5) dict lookups.
    # This will finish in microseconds.
    best_result = None
    for s in range(1, p - m + 1):  # s from 1 to 1004
        if time.time() > deadline:
            break
        # Check that all five values are non-zero quadratic residues
        all_qr = True
        roots = []
        for i in range(m):
            val = s + i
            if val not in root_of:
                all_qr = False
                break
            roots.append(root_of[val])
        if all_qr:
            best_result = {'start': s, 'roots': roots}
            break # Found minimum, can stop immediately

    if best_result:
        return best_result

    # Fallback: should not happen, but return a valid late run if we must.
    # Descending search as a safety net.
    for s in range(p - m, 0, -1):
        if time.time() > deadline:
            break
        roots = []
        ok = True
        for i in range(m):
            val = s + i
            if val not in root_of:
                ok = False
                break
            roots.append(root_of[val])
        if ok:
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run found')
# EVOLVE-BLOCK-END
