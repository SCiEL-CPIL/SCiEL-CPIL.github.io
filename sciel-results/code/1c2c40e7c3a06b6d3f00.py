def try_find(M, num_tries, deadline):
    coeffs = precompute_coeffs(walks)  # global
    for _ in range(num_tries):
        if time.time() > deadline: break
        a = [random.randrange(M) for _ in range(6)]
        ok = True
        for c in coeffs:
            s = 0
            for i in range(6):
                s += c[i]*a[i]
            if s % M == 0:
                ok = False
                break
        if ok:
            return a
    return None
