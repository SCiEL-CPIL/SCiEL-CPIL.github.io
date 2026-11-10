"""Structural redesign: multi-start randomized greedy with a direct
incremental C6 certificate and a repair-based local search.
Different design family from the simulated-annealing approach."""
import time
import random


def _edges_from_mask(mask, n):
    es = []
    for u in range(n):
        m = mask[u]
        while m:
            lb = m & (-m)
            m ^= lb
            v = lb.bit_length() - 1
            if u < v:
                es.append((u, v))
    return es


def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    n = 40

    adj = [0] * n

    def total_edges(mask):
        return sum(mask[u].bit_count() for u in range(n)) // 2

    def would_create_c6(u, v):
        """True if adding edge (u,v) (currently absent) creates a C6.
        Needs a length-5 path u-a-b-c-d-v in G: exists a in N(u), d in N(v)
        with a length-3 path a-b-c-d, internal vertices distinct from {u,v}."""
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        m = nu
        while m:
            la = m & (-m)
            m ^= la
            a = la.bit_length() - 1
            na = adj[a] & ~(1 << u) & ~(1 << v)  # candidate b
            if na == 0:
                continue
            md = nv
            while md:
                ld = md & (-md)
                md ^= ld
                d = ld.bit_length() - 1
                if d == a:
                    continue
                nc = adj[d] & ~(1 << u) & ~(1 << v)  # candidate c
                nc &= ~(1 << a)  # c != a
                mc = nc
                while mc:
                    lc = mc & (-mc)
                    mc ^= lc
                    c = lc.bit_length() - 1
                    bset = na & adj[c]
                    bset &= ~(1 << d)  # b != d
                    bset &= ~(1 << c)  # b != c
                    if bset:
                        return True
        return False

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def randomized_greedy():
        adj[:] = [0] * n
        cnt = 0
        order = list(all_pairs)
        random.shuffle(order)
        for (u, v) in order:
            if time.time() > deadline - 4:
                break
            if (1 << v) & adj[u]:
                continue
            if not would_create_c6(u, v):
                add_edge(u, v)
                cnt += 1
        return list(adj), cnt

    def repair_local_search(mask, time_limit):
        adj[:] = list(mask)
        best_mask = list(adj)
        best_cnt = total_edges(adj)
        while time.time() < time_limit:
            progress = False
            for (u, v) in all_pairs:
                if time.time() > time_limit:
                    break
                if (1 << v) & adj[u]:
                    continue
                if not would_create_c6(u, v):
                    add_edge(u, v)
                    progress = True
            cnt = total_edges(adj)
            if cnt > best_cnt:
                best_cnt = cnt
                best_mask = list(adj)
            if not progress:
                if time.time() > time_limit - 1:
                    break
                cur_edges = _edges_from_mask(adj, n)
                if not cur_edges:
                    break
                k = random.randint(1, 3)
                k = min(k, len(cur_edges))
                for (u, v) in random.sample(cur_edges, k):
                    remove_edge(u, v)
        return best_mask, best_cnt

    best_mask = [0] * n
    best_cnt = 0

    # Phase A: multiple randomized greedy starts
    num_starts = 6
    for s in range(num_starts):
        if time.time() > deadline - 30:
            break
        m, c = randomized_greedy()
        if c > best_cnt:
            best_cnt = c
            best_mask = m

    # Phase B: repair local search from best and perturbed copies
    remaining = deadline - time.time() - 5
    if remaining < 5:
        remaining = 5
    num_ls = max(1, int(remaining // 18))
    num_ls = min(num_ls, 6)
    for r in range(num_ls):
        if time.time() > deadline - 6:
            break
        if r == 0:
            start = list(best_mask)
        else:
            start = list(best_mask)
            ce = _edges_from_mask(start, n)
            if ce:
                for (u, v) in random.sample(ce, min(4, len(ce))):
                    start[u] &= ~(1 << v)
                    start[v] &= ~(1 << u)
        t_end = time.time() + (remaining / num_ls) * 0.9
        if t_end > deadline - 5:
            t_end = deadline - 5
        m, c = repair_local_search(start, t_end)
        if c > best_cnt:
            best_cnt = c
            best_mask = m

    # Phase C: final polish
    if time.time() < deadline - 3:
        m, c = repair_local_search(list(best_mask), deadline - 3)
        if c > best_cnt:
            best_cnt = c
            best_mask = m

    edges = []
    for (u, v) in _edges_from_mask(best_mask, n):
        edges.append([u, v])
    return edges
