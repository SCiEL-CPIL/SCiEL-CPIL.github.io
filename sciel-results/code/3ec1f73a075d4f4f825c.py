_P = 1009

def construct():
    # Precompute the smallest root for each non-zero quadratic residue mod 1009.
    # We need the smallest s in [1, 1004] such that s, s+1, s+2, s+3, s+4 are
    # all non-zero quadratic residues, and return the corresponding roots.
    root_map = {}
    is_residue = [False] * _P

    for x in range(1, _P):
        r = (x * x) % _P
        if r == 0:
            continue
        if r not in root_map:
            root_map[r] = x
            is_residue[r] = True

    # Scan for the smallest valid start
    for s in range(1, _P - 4):
        if (is_residue[s] and is_residue[s + 1] and is_residue[s + 2]
                and is_residue[s + 3] and is_residue[s + 4]):
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}

    raise RuntimeError("No run of 5 consecutive non-zero quadratic residues found")
