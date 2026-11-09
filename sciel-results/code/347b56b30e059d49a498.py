<<<<<<< SEARCH
def _greedy_construct(order):
    """Construct a dissociated set using a specific order of candidates."""
    S = []
    sums = {0}
    for x in order:
        v = _LCM // x
        is_valid = True
        for s in sums:
            if (s + v) in sums:
                is_valid = False
                break
        
        if is_valid:
            S.append(x)
            sums |= {s + v for s in sums}
    return S
=======
def _greedy_construct(order):
    """Construct a dissociated set using a specific order of candidates."""
    S = []
    sums = {0}
    for x in order:
        v = _LCM // x
        # Adding v is valid iff sums and {s+v for s in sums} are disjoint.
        # Equivalently, len(sums | {s+v for s in sums}) == 2*len(sums).
        shifted = {s + v for s in sums}
        if len(shifted) == len(sums) and not (sums & shifted):
            S.append(x)
            sums |= shifted
    return S
>>>>>>> REPLACE

