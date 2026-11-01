def _precompute_indices(n, m, p):
    a_indices = []
    for a in range(n * m):
        i = a // m
        j = a % m
        bws = []
        for b in range(m * p):
            jb = b // p
            kb = b % p
            if jb != j:
                continue
            for w in range(p * n):
                kw = w // n
                iw = w % n
                if kw == kb and iw == i:
                    bws.append((b, w, 1.0))
                else:
                    bws.append((b, w, 0.0))
        a_indices.append(bws)
    return a_indices
