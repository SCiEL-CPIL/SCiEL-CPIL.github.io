import time
import random
from math import gcd

def construct():
    """
    Largest subset of [1,128] with no three distinct elements whose three
    pairwise gcds are equal.  Hypergraph-independence search with multi-kick
    local search.  Returns a list of ints.
    """
    start = time.time()
    deadline = start + 84.0          # 90s kill, keep 6s margin
    N = 128
    ALL = (1 << (N + 1)) - 1
    ALL ^= 1                          # clear bit 0

    # ---------- precompute gcd and bad-triple conflict structure ----------
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    conflict = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gab == gcd_mat[a][c] == gcd_mat[b][c]:
                    conflict[a].append((b, c))
                    conflict[b].append((a, c))
                    conflict[c].append((a, b))

    # bitmask of the "partner" element for each conflict pair
    pair_partner = {}
    for x in range(1, N + 1):
        for a, b in conflict[x]:
            pair_partner[a] = b

    # ---------- feasibility helpers (bitmask based) ----------
    def can_add(x, s):
        for a, b in conflict[x]:
            if (s >> a) & 1 and (s >> b) & 1:
                return False
        return True

    def greedy_fill(s):
        """Repeatedly add any element that can be added (natural order)."""
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if (s >> x) & 1:
                    continue
                ok = True
                for a, b in conflict[x]:
                    if (s >> a) & 1 and (s >> b) & 1:
                        ok = False
                        break
                if ok:
                    s |= (1 << x)
                    changed = True
                    break
        return s

    # ---------- local search with multi-removal kicks ----------
    def local_search(s, budget):
        best = s
        best_sz = s.bit_count()
        while time.time() < budget:
            s = greedy_fill(s)
            sz = s.bit_count()
            if sz > best_sz:
                best = s
                best_sz = sz
                continue
            if sz == 0:
                break
            # local maximum: try kicks of 1..3 removals
            elems = [i for i in range(1, N + 1) if (s >> i) & 1]
            improved = False
            # 1-kick
            for k in elems:
                if time.time() > budget:
                    break
                ns = greedy_fill(s & ~(1 << k))
                if ns.bit_count() > best_sz:
                    best = ns
                    best_sz = ns.bit_count()
                    s = ns
                    improved = True
                    break
            if improved:
                continue
            # 2-kick
            for i in range(len(elems)):
                if time.time() > budget:
                    break
                a = elems[i]
                for j in range(i + 1, len(elems)):
                    b = elems[j]
                    ns = greedy_fill(s & ~(1 << a) & ~(1 << b))
                    if ns.bit_count() > best_sz:
                        best = ns
                        best_sz = ns.bit_count()
                        s = ns
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # 3-kick
            for i in range(len(elems)):
                if time.time() > budget:
                    break
                a = elems[i]
                for j in range(i + 1, len(elems)):
                    b = elems[j]
                    for kk in range(j + 1, len(elems)):
                        c = elems[kk]
                        ns = greedy_fill(s & ~(1 << a) & ~(1 << b) & ~(1 << c))
                        if ns.bit_count() > best_sz:
                            best = ns
                            best_sz = ns.bit_count()
                            s = ns
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    def kick_refill(s, k):
        """Remove k random elements, refill, return best of the two."""
        elems = [i for i in range(1, N + 1) if (s >> i) & 1]
        if len(elems) < k:
            return s
        rem = random.sample(elems, k)
        ns = s
        for r in rem:
            ns &= ~(1 << r)
        ns = greedy_fill(ns)
        return ns if ns.bit_count() > s.bit_count() else s

    # ---------- build strategies ----------
    def degree_build():
        deg = [len(conflict[x]) for x in range(N + 1)]
        s = 0
        while True:
            best_x = -1
            best_d = 1 << 60
            for x in range(1, N + 1):
                if (s >> x) & 1:
                    continue
                if can_add(x, s) and deg[x] < best_d:
                    best_d = deg[x]
                    best_x = x
            if best_x < 0:
                break
            s |= (1 << best_x)
        return s

    def order_build(order):
        s = 0
        for x in order:
            if can_add(x, s):
                s |= (1 << x)
        return s

    best = 0
    best_sz = 0
    def upd(s):
        nonlocal best, best_sz
        if s.bit_count() > best_sz:
            best = s
            best_sz = s.bit_count()

    # deterministic starts
    for s in (degree_build(), order_build(range(1, N + 1)),
              order_build(range(N, 0, -1))):
        upd(s)

    # random restarts with local search, then random kicks
    while time.time() < deadline - 3.0:
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = order_build(order)
        s = local_search(s, deadline - 2.0)
        upd(s)
        # extra random kicks on the local optimum
        for _ in range(8):
            if time.time() > deadline - 2.0:
                break
            s2 = kick_refill(s, random.choice((1, 2, 3)))
            if s2.bit_count() > s.bit_count():
                s = s2
            upd(s)

    # final polish: systematic 1-kick sweep on best
    if time.time() < deadline - 1.0:
        s = best
        for u in range(1, N + 1):
            if time.time() > deadline - 1.0:
                break
            if (s >> u) & 1:
                ns = greedy_fill(s & ~(1 << u))
                if ns.bit_count() > s.bit_count():
                    s = ns
        upd(s)

    res = [i for i in range(1, N + 1) if (best >> i) & 1]
    if not res:
        res = [1]
    return res
