def _try_extend(moduli, assign, new_idx):
    n = len(moduli)
    m = moduli[new_idx]
    allowed = [True] * m
    for j in range(n):
        g = gcd(m, moduli[j])
        rj = assign[j] % g
        for r in range(rj, m, g):
            allowed[r] = False
    for r in range(m):
        if allowed[r]:
            return r
    return None
