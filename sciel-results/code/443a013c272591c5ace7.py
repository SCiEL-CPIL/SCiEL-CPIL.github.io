def _is_dissociated(S):
    if len(S) <= 1:
        return True
    L = 1
    for s in S:
        L = _lcm(L, s)
    weights = [L // s for s in S]
    
    seen = {0}
    sums = [0]
    for w in weights:
        new_sums = [x + w for x in sums]
        for ns in new_sums:
            if ns in seen:
                return False
        seen.update(new_sums)
        sums.extend(new_sums)
    return True
