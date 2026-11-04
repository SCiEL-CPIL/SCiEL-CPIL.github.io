<<<<<<< SEARCH
    # ---------------- main ----------------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    # k == 3: try Ramanujan graphs (smaller), else LUW.
    best = None
    best_n = float('inf')

    # Try primes: Ramanujan cubic graphs have n = q(q+1)/2 vertices.
    # For girth >= 16, we need q large enough. Known: q=19 gives girth 8,
    # q=31 gives girth 8, q=79 gives girth 10, q=151 gives girth 12,
    # q=211 gives girth 14, q=433 gives girth 16 (n=93606).
    # But our construction may not achieve optimal girth for all q.
    # Try a range of primes and check girth.
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]
    for q in primes:
        if remaining() < 10:
            break
        # Moore bound for k=3, g=16: n >= 2 + 3*(1 + 2 + 4 + 8 + 16 + 32) = 2 + 3*63 = 191
        # Ramanujan n = q(q+1)/2, so q >= ~20 for n >= 191
        if q * (q + 1) // 2 < 191:
            continue
        res = build_ramanujan(q)
        if res is None:
            continue
        n, edges = res
        if n >= best_n:
            continue
        gc = check_girth(n, edges, g, deadline - 8)
        if gc == -1:
            continue
        if gc >= g:
            best, best_n = (n, edges), n
            if n <= 3000:  # excellent, stop
                break

    if best is not None:
        return best

    # fall back to LUW (guaranteed valid)
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
=======
    # ---------------- main ----------------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    # k == 3: try Ramanujan graphs (smaller), else LUW + lift.
    best = None
    best_n = float('inf')

    # Try primes: Ramanujan cubic graphs have n = q(q+1)/2 vertices.
    # For girth >= 16, need q large. Try moderate primes and also lift.
    primes = [19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433]
    for q in primes:
        if remaining() < 15:
            break
        if q * (q + 1) // 2 < 191:
            continue
        res = build_ramanujan(q)
        if res is None:
            continue
        n, edges = res
        if n >= best_n:
            continue
        gc = check_girth(n, edges, g, deadline - 10)
        if gc == -1:
            continue
        if gc >= g:
            best, best_n = (n, edges), n
            if n <= 3000:
                break

    # Also try girth-8 lift on small Ramanujan graphs
    if remaining() > 30:
        for q in [19, 23, 29, 31, 37, 41, 43, 47, 53]:
            if remaining() < 20:
                break
            res = build_ramanujan(q)
            if res is None:
                continue
            n_base, edges_base = res
            gc_base = check_girth(n_base, edges_base, 8, deadline - 20)
            if gc_base == -1 or gc_base < 8:
                continue
            for m_lift in [4, 3]:
                if remaining() < 15:
                    break
                lifted = girth8_lift_optimal(n_base, edges_base, m_lift, deadline - 12)
                if lifted is not None:
                    n_lift, edges_lift = lifted
                    if n_lift < best_n:
                        best, best_n = (n_lift, edges_lift), n_lift
                        if n_lift <= 5000:
                            break

    if best is not None:
        return best

    # fall back to LUW (guaranteed valid)
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
>>>>>>> REPLACE
