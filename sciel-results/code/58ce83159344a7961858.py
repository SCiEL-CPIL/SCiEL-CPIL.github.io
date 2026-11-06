import time
import random
import itertools

N = 20
K = 5           # number of groups
G = 4           # vertices per group

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def has_k33_fast(adj):
        """Return True iff no three vertices have >= 3 common neighbors."""
        for i, j, k in itertools.combinations(range(N), 3):
            if (adj[i] & adj[j] & adj[k]).bit_count() >= 3:
                return False
        return True

    def is_add_valid(adj, u, v):
        """Return True iff adding edge (u,v) keeps the graph K3,3-free."""
        nu = adj[u]
        nv = adj[v]
        # Triples containing u: i<j in N(v)\{u} with >=3 common neighbors
        cu = nv & ~(1 << u)
        lst = [x for x in range(N) if (cu >> x) & 1]
        nu_new = nu | (1 << v)
        for a in range(len(lst)):
            ia = lst[a]
            ai = adj[ia]
            for b in range(a + 1, len(lst)):
                ib = lst[b]
                if (nu_new & ai & adj[ib]).bit_count() >= 3:
                    return False
        # Triples containing v
        cv = nu & ~(1 << v)
        lst = [x for x in range(N) if (cv >> x) & 1]
        nv_new = nv | (1 << u)
        for a in range(len(lst)):
            ia = lst[a]
            ai = adj[ia]
            for b in range(a + 1, len(lst)):
                ib = lst[b]
                if (nv_new & ai & adj[ib]).bit_count() >= 3:
                    return False
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def non_edges(adj):
        res = []
        for i in range(N):
            mi = adj[i]
            for j in range(i + 1, N):
                if not (mi >> j) & 1:
                    res.append((i, j))
        return res

    def all_edges(adj):
        res = []
        for i in range(N):
            mi = adj[i]
            for j in range(i + 1, N):
                if (mi >> j) & 1:
                    res.append((i, j))
        return res

    def greedy_add(adj, deadline):
        """Repeatedly add any valid non-edge (random order) until none remain."""
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            ne = non_edges(adj)
            random.shuffle(ne)
            for u, v in ne:
                if is_add_valid(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
        return adj

    def local_search(start, deadline):
        """Greedy add, then random 1-edge swaps (remove one, add one) to improve."""
        adj = start[:]
        best = adj[:]
        best_e = count_edges(adj)

        def snapshot():
            nonlocal best, best_e
            e = count_edges(adj)
            if e > best_e:
                best_e = e
                best = adj[:]

        greedy_add(adj, deadline)
        snapshot()

        while time.time() < deadline:
            moved = False
            # 1) try to add a non-edge
            ne = non_edges(adj)
            if ne:
                random.shuffle(ne)
                for u, v in ne:
                    if is_add_valid(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        moved = True
                        break
            # 2) try a 1-edge swap
            if not moved:
                ae = all_edges(adj)
                ne = non_edges(adj)
                if ae and ne:
                    for _ in range(60):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(ae)
                        c, d = random.choice(ne)
                        m = adj[:]
                        m[a] &= ~(1 << b)
                        m[b] &= ~(1 << a)
                        if is_add_valid(m, c, d):
                            m[c] |= (1 << d)
                            m[d] |= (1 << c)
                            if has_k33_fast(m):
                                adj = m
                                moved = True
                                break
            snapshot()
            if not moved:
                break
        return best, best_e

    def direct_construct():
        """Deterministic K3,3-free graph from the 5x4 group construction."""
        adj = [0] * N
        cycle = [0, 1, 2, 3]
        diag = [(0, 2), (1, 3)]
        for g in range(K):
            base = g * G
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                u, v = base + a, base + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            for a, b in diag:
                u, v = base + a, base + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        for g in range(K):
            for h in range(g + 1, K):
                for a in range(G):
                    for b in range(G):
                        u, v = g * G + a, h * G + b
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    def bitmasks_to_edges(adj):
        edges = []
        for i in range(N):
            mi = adj[i]
            for j in range(i + 1, N):
                if (mi >> j) & 1:
                    edges.append([i, j])
        return edges

    # Start with the deterministic 5x4 group construction
    base = direct_construct()
    if not has_k33_fast(base):
        # defensive fallback: rebuild the core without inter-group edges
        base = [0] * N
        cycle = [0, 1, 2, 3]
        diag = [(0, 2), (1, 3)]
        for g in range(K):
            gb = g * G
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                u, v = gb + a, gb + b
                base[u] |= (1 << v)
                base[v] |= (1 << u)
            for a, b in diag:
                u, v = gb + a, gb + b
                base[u] |= (1 << v)
                base[v] |= (1 << u)
        # re-add inter-group one by one only while valid
        for g in range(K):
            for h in range(g + 1, K):
                cand = base[:]
                ok = True
                for a in range(G):
                    for b in range(G):
                        u, v = g * G + a, h * G + b
                        if not is_add_valid(cand, u, v):
                            ok = False
                            break
                        cand[u] |= (1 << v)
                        cand[v] |= (1 << u)
                    if not ok:
                        break
                if ok:
                    base = cand

    best = base[:]
    best_e = count_edges(base)

    if has_k33_fast(base):
        improved, e = local_search(base[:], deadline)
        if e > best_e:
            best_e = e
            best = improved

    if has_k33_fast(best):
        for r in range(8):
            if time.time() > deadline - 5:
                break
            cand = [0] * N
            p = 0.15 + 0.05 * (r % 4)
            for i in range(N):
                for j in range(i + 1, N):
                    if random.random() < p:
                        cand[i] |= (1 << j)
                        cand[j] |= (1 << i)
            att = 0
            while not has_k33_fast(cand) and att < 2000:
                ae = all_edges(cand)
                if not ae:
                    break
                u, v = random.choice(ae)
                cand[u] &= ~(1 << v)
                cand[v] &= ~(1 << u)
                att += 1
            if not has_k33_fast(cand):
                continue
            improved, e = local_search(cand, deadline)
            if e > best_e:
                best_e = e
                best = improved

    final = greedy_add(best[:], deadline)
    fe = count_edges(final)
    if fe > best_e:
        best_e = fe
        best = final

    return bitmasks_to_edges(best)
