<<<<<<< SEARCH
    # a small list of (a, b) generator pairs (kept distinct, nonzero, a != -b)
    pairs = []
    for a in range(1, 8):
        for b in range(1, 8):
            if b != a and b != (-a) % 10**9 and a != 0 and b != 0:
                pairs.append((a, b))
    # dedupe up to a reasonable count
    seen = set()
    pairs2 = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            pairs2.append(p)
    pairs = pairs2

    q = lo_q
    while q <= hi_q and _time_left() > 0:
        q = _next_prime(q)
        for (a, b) in pairs:
            if q <= max(a, b):
                continue
            if _time_left() <= 0:
                break
            n, adj, edges = _cayley_zq_z2(q, a % q, b % q)
            if n < best_n and _girth_ok(adj, n, g):
                best_n, best_edges = n, edges
                # found a smaller valid one; continue scanning to shrink further
                # but we can stop once we hit the Moore bound (optimal in family)
                if 2 * q >= moore and n <= moore:
                    return n, edges
        q += 1

    return best_n, best_edges
=======
    # Generate candidate (a, b) pairs for various q values
    # We'll generate pairs on-the-fly for each q to maximize diversity

    q = lo_q
    while q <= hi_q and _time_left() > 0:
        # Try both prime and composite q
        candidates = [q]
        if _is_prime(q):
            candidates.append(q + 1) if q + 1 <= hi_q else None
        # Try a few nearby values
        for dq in [0, 1, 2, 3]:
            if q + dq <= hi_q and _time_left() > 0:
                qq = q + dq
                # Generate (a, b) pairs for this qq
                max_gen = min(12, qq // 2)
                for a in range(1, max_gen + 1):
                    for b in range(a + 1, max_gen + 1):
                        if _time_left() <= 0:
                            break
                        # Check generators are distinct and valid
                        if a == (-b) % qq:
                            continue
                        n, adj, edges = _cayley_zq_z2(qq, a, b)
                        if n is None:
                            continue
                        if n < best_n and _girth_ok(adj, n, g):
                            best_n, best_edges = n, edges
                            if 2 * qq >= moore and n <= moore:
                                return n, edges
        q += 1

    return best_n, best_edges
>>>>>>> REPLACE

