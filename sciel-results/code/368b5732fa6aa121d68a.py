"""Find minimum s such that s, s+1, s+2, s+3, s+4 are all non-zero quadratic
residues mod 1009, each certified by a root in [1,1008]."""
import time

def construct():
    p = 1009
    m = 5
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Precompute: for each non-zero residue r in [1, p-1], store a root x in [1, p-1]
    # such that x^2 % p == r. Use a list for O(1) direct indexing.
    root_of = [0] * (p + 5)  # extra slots to avoid bounds issues
    is_res = [False] * (p + 5)

    for x in range(1, p):
        if time.time() > deadline:
            break
        r = (x * x) % p
        if r != 0 and not is_res[r]:
            root_of[r] = x
            is_res[r] = True

    # Search for the minimum s in [1, 1004] such that
    # s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues.
    # Since p=1009, s+4 <= 1008, so no wraparound needed.
    for s in range(1, p - m + 1):  # s from 1 to 1004
        if time.time() > deadline:
            break
        # Check that all five values are non-zero quadratic residues using boolean array
        if (is_res[s] and is_res[s + 1] and is_res[s + 2]
                and is_res[s + 3] and is_res[s + 4]):
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    # Fallback: descending search as a safety net
    for s in range(p - m, 0, -1):
        if time.time() > deadline:
            break
        if (is_res[s] and is_res[s + 1] and is_res[s + 2]
                and is_res[s + 3] and is_res[s + 4]):
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run found')
