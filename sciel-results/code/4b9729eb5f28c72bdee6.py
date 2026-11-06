"""C4-free construction: polarity-graph seed + bitmask local search + fast C4 check."""
import time
import random
import numpy as np


def _prime_power(n):
    """Largest prime power <= n (n>=2)."""
    if n < 2:
        return 1
    best = 1
    for p in range(2, n + 1):
        # primality test
        is_p = True
        for d in range(2, int(p ** 0.5) + 1):
            if p % d == 0:
                is_p = False
                break
        if not is_p:
            continue
        pk = p
        while pk <= n:
            best = pk
            pk *= p
    return best


def _polarity_graph(q, n):
    """Build a C4-free polarity graph on n vertices with uniform degree q+1.

    Uses the standard polarity of the projective plane of order q (points are
    1-dim subspaces of GF(q)^3). A point is adjacent to a line iff the line lies
    in the polar of the point; the diagonal of the polarity matrix is zero, so
    the graph is C4-free. We take the first n points as vertices.
    """
    # Build GF(q)
    if q == 2:
        # GF(2) = {0,1}
        gmul = lambda a, b: (a & b)
        gadd = lambda a, b: (a ^ b)
        ginv = lambda a: a
    else:
        # q is a prime power; for prime q use integers mod q. For prime powers
        # we fall back to a simple representation only when prime; otherwise
        # (rare) we still handle it with a generic finite field via primitive
        # element search.
        # Determine if q is prime
        is_prime = True
        for d in range(2, int(q ** 0.5) + 1):
            if q % d == 0:
                is_prime = False
                break
        if is_prime:
            gmul = lambda a, b: (a * b) % q
            gadd = lambda a, b: (a + b) % q
            ginv = lambda a: pow(a, q - 2, q)
        else:
            # Build GF(p^k) via primitive polynomial
            p = 2
            while p * p <= q:
                if q % p == 0:
                    break
                p += 1
            k = 0
            tmp = q
            while tmp % p == 0:
                tmp //= p
                k += 1
            # Find a primitive polynomial of degree k over GF(p)
            def poly_eval(coeffs, x):
                # coeffs: list [c0..ck], evaluate at element x (int code)
                # Represent GF(p^k) elements as integers 0..p^k-1 (base-p digits)
                # This is a fallback; correctness for non-prime q is best-effort.
                acc = 0
                for c in reversed(coeffs):
                    acc = gmul_fallback(acc, x) + c
                    acc = acc % (p ** k)
                return acc
            # We'll implement GF(p^k) with integer representation via a
            # multiplication table built from a primitive polynomial.
            # Find primitive polynomial
            import itertools
            found = None
            for c0 in range(p):
                for rest in itertools.product(range(p), repeat=k - 1):
                    coeffs = [c0] + list(rest)  # c0..c_{k-1}, monic leading 1
                    # skip if leading coefficient zero (monic means leading 1)
                    # Build multiplication table
                    def mul_table(coeffs):
                        # element x: base-p digits d0..d_{k-1}
                        # x = sum d_i alpha^i
                        # alpha^k = - (c0 + c1 alpha + ... + c_{k-1} alpha^{k-1})
                        def digit(x, i):
                            return (x // (p ** i)) % p
                        def to_val(digits):
                            v = 0
                            for i, d in enumerate(digits):
                                v += d * (p ** i)
                            return v
                        def reduce_poly(d):
                            # d: list of coefficients length up to 2k-1
                            # reduce alpha^k using the relation
                            # alpha^k = -(c0 + c1 alpha + ... + c_{k-1} alpha^{k-1})
                            while len(d) > k:
                                # take highest term
                                hi = len(d) - 1
                                coeff = d[hi]
                                # alpha^hi = alpha^{hi-k} * alpha^k
                                # alpha^k = sum_{j=0}^{k-1} (-c_j) alpha^j
                                for j in range(k):
                                    d[hi - k + j] = (d[hi - k + j] + coeff * (-coeffs[j])) % p
                                d[hi] = 0
                            return d
                        def add(a, b):
                            return [(a[i] + b[i]) % p for i in range(k)]
                        def scale(a, c):
                            return [(a[i] * c) % p for i in range(k)]
                        def mul(a, b):
                            # a,b: list length k
                            # product polynomial length 2k-1
                            prod = [0] * (2 * k - 1)
                            for i in range(k):
                                if a[i] == 0:
                                    continue
                                for j in range(k):
                                    if b[j] == 0:
                                        continue
                                    prod[i + j] = (prod[i + j] + a[i] * b[j]) % p
                            # reduce
                            for hi in range(2 * k - 2, k - 1, -1):
                                if prod[hi] != 0:
                                    for j in range(k):
                                        prod[hi - k + j] = (prod[hi - k + j] + prod[hi] * (-coeffs[j])) % p
                                    prod[hi] = 0
                            return prod[:k]
                        # Build table
                        table = [[0] * (p ** k) for _ in range(p ** k)]
                        elems = list(range(p ** k))
                        def to_digits(e):
                            return [digit(e, i) for i in range(k)]
                        def to_val(d):
                            return to_val(d)
                        for a in elems:
                            da = to_digits(a)
                            for b in elems:
                                db = to_digits(b)
                                prod = mul(da, db)
                                table[a][b] = to_val(prod)
                        return table
                    t = mul_table(coeffs)
                    # Check primitive: find element with order p^k - 1
                    order_target = p ** k - 1
                    primitive = False
                    for a in range(1, p ** k):
                        x = a
                        for _ in range(order_target - 1):
                            x = t[x][a]
                        if x == 1:
                            # check no smaller order
                            ok = True
                            x2 = a
                            for d in range(2, order_target + 1):
                                if order_target % d == 0:
                                    x2 = a
                                    for _ in range(d - 1):
                                        x2 = t[x2][a]
                                    if x2 == 1:
                                        ok = False
                                        break
                            if ok:
                                primitive = True
                                break
                    if primitive:
                        found = (coeffs, t)
                        break
                if found:
                    break
            if found is None:
                # Should not happen; fall back to a random C4-free seed
                return None
            coeffs, gtable = found
            gmul = lambda a, b: gtable[a][b]
            gadd = lambda a, b: (a + b) % (p ** k)
            # inverse via table
            inv_table = [0] * (p ** k)
            for a in range(1, p ** k):
                for b in range(1, p ** k):
                    if gtable[a][b] == 1:
                        inv_table[a] = b
                        break
            ginv = lambda a: inv_table[a]

    # Enumerate points: 1-dim subspaces of GF(q)^3, in lex order, take first n
    def points():
        pts = []
        for a in range(q):
            for b in range(q):
                pts.append((a, b, 1))
        for b in range(q):
            pts.append((0, b, 1))
        pts.append((0, 0, 1))
        return pts

    # Actually standard enumeration: points represented as (x,y,z) not all zero,
    # modulo scalar. We generate a canonical representative for each line through origin.
    def all_points():
        reps = []
        seen = set()
        for z in range(q):
            for y in range(q):
                for x in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # canonical: find first nonzero coordinate, scale to make it 1
                    if x != 0:
                        invx = ginv(x)
                        rep = (1, gmul(y, invx), gmul(z, invx))
                    elif y != 0:
                        invy = ginv(y)
                        rep = (0, 1, gmul(z, invy))
                    else:
                        rep = (0, 0, 1)
                    if rep not in seen:
                        seen.add(rep)
                        reps.append(rep)
        return reps

    pts = all_points()
    # Take first n points (need n <= q^2+q+1)
    if n > len(pts):
        n = len(pts)
    sel = pts[:n]

    # Polarity: point p=(x,y,z) polar line: x*X + y*Y + z*Z = 0
    # Point p is adjacent to point p'=(x',y',z') iff p' lies on polar of p:
    #   x*x' + y*y' + z*z' = 0
    adj = [0] * n
    for i in range(n):
        x, y, z = sel[i]
        for j in range(i + 1, n):
            x2, y2, z2 = sel[j]
            s = gadd(gadd(gmul(x, x2), gmul(y, y2)), gmul(z, z2))
            if s == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _is_c4_free_fast(adj, n):
    """Fast C4 check via (A^2)[i][j] for i!=j: count common neighbors."""
    A = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        a = adj[i]
        # set bits
        tmp = a
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            A[i][idx] = 1
            tmp ^= lsb
    A2 = A @ A
    # For i != j, A2[i][j] = number of common neighbors. C4 iff >= 2.
    # Diagonal A2[i][i] = degree, ignore.
    np.fill_diagonal(A2, 0)
    return not (A2 >= 2).any()


def _is_c4_free(adj, n):
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai & adj[j]).bit_count() >= 2:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def build_edges(adj, n):
        edges = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    edges.append([i, j])
        return edges

    def min_deg(adj, n):
        m = n
        for i in range(n):
            d = adj[i].bit_count()
            if d < m:
                m = d
        return m

    def local_search(adj, n, target, budget):
        """Add edges to raise min degree while preserving C4-freeness."""
        degs = [adj[i].bit_count() for i in range(n)]
        mask_all = (1 << n) - 1
        for _ in range(200):
            if elapsed() > budget:
                break
            if min(degs) >= target:
                break
            progressed = False
            low = [i for i in range(n) if degs[i] <= min(degs) + 1]
            random.shuffle(low)
            for u in low:
                if degs[u] >= target:
                    continue
                non = (~adj[u]) & mask_all
                non &= ~(1 << u)
                if not non:
                    continue
                tmp = non
                cands = []
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    cands.append(idx)
                    tmp ^= lsb
                random.shuffle(cands)
                for v in cands:
                    if (adj[u] & adj[v]).bit_count() < 2:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degs[u] += 1
                        degs[v] += 1
                        progressed = True
                        if degs[u] >= target:
                            break
                if degs[u] >= target:
                    continue
            if not progressed:
                break
        return adj

    def try_n(n, budget):
        target = n - 39
        if target < 1:
            target = 1
        best = None
        best_md = -1

        # Seed 1: polarity graph
        try:
            pg = _polarity_graph(_prime_power(n - 1), n)
            if pg is not None and _is_c4_free_fast(pg, n):
                md = min_deg(pg, n)
                if md > best_md:
                    best_md = md
                    best = pg[:]
                if md < target and elapsed() < budget:
                    ls = local_search(pg, n, target, budget)
                    if _is_c4_free(ls, n):
                        md2 = min_deg(ls, n)
                        if md2 > best_md:
                            best_md = md2
                            best = ls[:]
                        if md2 >= target:
                            return n, build_edges(ls, n)
        except Exception:
            pass

        # Seed 2+: random graphs with repair + local search
        ps = [target / n, (target + 1) / n, (target + 2) / n, 0.55]
        ps = [p for p in ps if 0 < p < 0.9]
        if not ps:
            ps = [0.5]

        for p in ps:
            for attempt in range(3):
                if elapsed() > budget:
                    break
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)

                # Repair: remove edges to break C4s
                for _ in range(150):
                    if elapsed() > budget:
                        break
                    # find a C4
                    found = False
                    for i in range(n):
                        ai = adj[i]
                        for j in range(i + 1, n):
                            common = ai & adj[j]
                            if common.bit_count() >= 2:
                                tmp = common
                                c1 = None
                                while tmp:
                                    lsb = tmp & -tmp
                                    idx = lsb.bit_length() - 1
                                    if c1 is None:
                                        c1 = idx
                                    else:
                                        adj[i] &= ~(1 << c1)
                                        adj[c1] &= ~(1 << i)
                                        found = True
                                        break
                                    tmp ^= lsb
                                if found:
                                    break
                        if found:
                            break
                    if not found:
                        break

                if not _is_c4_free(adj, n):
                    continue

                md = min_deg(adj, n)
                if md > best_md:
                    best_md = md
                    best = adj[:]

                if md < target and elapsed() < budget:
                    ls = local_search(adj, n, target, budget)
                    if _is_c4_free(ls, n):
                        md2 = min_deg(ls, n)
                        if md2 > best_md:
                            best_md = md2
                            best = ls[:]
                        if md2 >= target:
                            return n, build_edges(ls, n)

        if best is not None and best_md >= target:
            return n, build_edges(best, n)
        return None

    # Try from largest n down, allocating more time to larger n
    ns = [47, 46, 45, 44, 43, 42, 41, 40]
    weights = [1.5, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
    total_w = sum(weights)
    alloc = [int((deadline - 2) * w / total_w) for w in weights]
    alloc = [max(5, a) for a in alloc]
    s = sum(alloc)
    if s > deadline - 2:
        scale = (deadline - 2) / s
        alloc = [max(3, int(a * scale)) for a in alloc]

    for n, b in zip(ns, alloc):
        if elapsed() > deadline - 1:
            break
        res = try_n(n, deadline - 1)
        if res is not None:
            return {'n': res[0], 'edges': res[1]}

    # Fallback: trivial graph
    return {'n': 39, 'edges': []}
