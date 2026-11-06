import time
import random

def construct():
    t0 = time.monotonic()
    DL = t0 + 140.0          # safety margin under 150s
    N = 40

    def make_adj(edges):
        adj = [0] * N
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    def neigh(adj):
        return [[i for i in range(N) if (adj[x] >> i) & 1] for x in range(N)]

    def union_nb(adj, mask):
        r = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            r |= adj[i]
            m ^= lsb
        return r

    def quick_reach5(adj, u, v):
        ban = (1 << u) | (1 << v)
        cur = 1 << u
        for s in range(5):
            cur = union_nb(adj, cur) & ~ban
            if s == 4:
                return bool(cur & (1 << v))
        return False

    def detailed_c6(adj, nb, u, v):
        for b in range(N):
            if b == u or b == v:
                continue
            ab = adj[b]
            A = [x for x in nb[u] if x != v and x != b and (ab >> x) & 1]
            if not A:
                continue
            D = [x for x in nb[v] if x != u and x != b and (ab >> x) & 1]
            if not D:
                continue
            for a in A:
                for d in D:
                    if a == d:
                        continue
                    for c in nb[b]:
                        if c == u or c == v or c == a or c == b or c == d:
                            continue
                        if (adj[c] >> d) & 1:
                            return True
        return False

    def creates_c6(adj, nb, u, v):
        if not quick_reach5(adj, u, v):
            return False
        return detailed_c6(adj, nb, u, v)

    def edge_list(adj):
        return [[i, j] for i in range(N) for j in range(i + 1, N) if (adj[i] >> j) & 1]

    def count(adj):
        return sum(adj[i].bit_count() for i in range(N)) // 2

    def greedy_add(adj, nb):
        changed = True
        while changed and time.monotonic() < DL - 3:
            changed = False
            for i in range(N):
                if time.monotonic() > DL - 3:
                    return adj
                for j in range(i + 1, N):
                    if (adj[i] >> j) & 1:
                        continue
                    if not creates_c6(adj, nb, i, j):
                        adj[i] |= 1 << j
                        adj[j] |= 1 << i
                        nb[i].append(j)
                        nb[j].append(i)
                        changed = True
        return adj

    def improve(adj):
        nb = neigh(adj)
        greedy_add(adj, nb)
        best_n = count(adj)
        while time.monotonic() < DL - 5:
            improved = False
            edges = edge_list(adj)
            random.shuffle(edges)
            attempts = 0
            for (e1, e2) in edges:
                if time.monotonic() > DL - 5:
                    break
                if attempts >= 150:
                    break
                attempts += 1
                # remove edge
                adj[e1] ^= 1 << e2
                adj[e2] ^= 1 << e1
                nb2 = neigh(adj)
                greedy_add(adj, nb2)
                n = count(adj)
                if n > best_n:
                    best_n = n
                    improved = True
                    nb = nb2
                else:
                    # revert (edge was safe before)
                    adj[e1] |= 1 << e2
                    adj[e2] |= 1 << e1
                    nb = neigh(adj)
            if not improved:
                break
        return adj, best_n

    def base_k5():
        edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.append([5 * k + a, 5 * k + b])
        return edges

    def base_random(p):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < p:
                    edges.append([i, j])
        return edges

    def prune_c6(edges):
        adj = make_adj(edges)
        nb = neigh(adj)
        changed = True
        while changed and time.monotonic() < DL - 8:
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    if (adj[i] >> j) & 1 and creates_c6(adj, nb, i, j):
                        adj[i] ^= 1 << j
                        adj[j] ^= 1 << i
                        nb = neigh(adj)
                        changed = True
        return adj

    best_adj = None
    best_n = -1
    random.seed(987654321)

    # Seed 1: K5 union (guaranteed C6-free, 80 edges)
    adj = make_adj(base_k5())
    adj, n = improve(adj)
    if n > best_n:
        best_n = n
        best_adj = [a[:] for a in adj]

    # Seeds 2..: random pruned graphs
    for p in [0.45, 0.55, 0.65, 0.5]:
        if time.monotonic() > DL - 12:
            break
        adj = prune_c6(base_random(p))
        adj, n = improve(adj)
        if n > best_n:
            best_n = n
            best_adj = [a[:] for a in adj]

    # Final guarantee: strip any edge that participates in a C6
    adj = best_adj
    nb = neigh(adj)
    for i in range(N):
        for j in range(i + 1, N):
            if (adj[i] >> j) & 1 and creates_c6(adj, nb, i, j):
                adj[i] ^= 1 << j
                adj[j] ^= 1 << i
    return edge_list(adj)
