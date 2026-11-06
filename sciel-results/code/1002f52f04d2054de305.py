import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute adjacency lists from bitmask
    def neighbors(adj, v):
        mask = adj[v]
        out = []
        while mask:
            lsb = mask & (-mask)
            out.append(lsb.bit_length() - 1)
            mask ^= lsb
        return out

    # Fast K2,3 check: adding (u,v) is valid iff
    #  (a) |N(u) ∩ N(v)| <= 2  (u,v don't share 3 neighbors)
    #  (b) for every w in N(u)\{v}: |N(v) ∩ N(w)| <= 1  (else {v,w} share 3 with u)
    #  (c) for every w in N(v)\{u}: |N(u) ∩ N(w)| <= 1
    def is_k23_free(adj, u, v):
        au = adj[u]
        av = adj[v]
        if (au & av).bit_count() > 2:
            return False
        # w in N(u) not v
        for w in neighbors(adj, u):
            if w == v:
                continue
            if (av & adj[w]).bit_count() > 1:
                return False
        # w in N(v) not u
        for w in neighbors(adj, v):
            if w == u:
                continue
            if (au & adj[w]).bit_count() > 1:
                return False
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            v = u + 1
            while v < n:
                if (mask >> v) & 1:
                    edges.append([u, v])
                v += 1
        return edges

    def greedy_build(seed=None):
        rng = random.Random(seed)
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for (u, v) in all_edges:
            if is_k23_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, time_budget=5.0, rng=None):
        if rng is None:
            rng = random.Random()
        adj = list(adj)
        search_start = time.time()

        present = set()
        missing = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    present.add((u, v))
                else:
                    missing.add((u, v))

        improved = True
        while improved:
            if time.time() - search_start > time_budget:
                break
            improved = False

            # Add any feasible missing edge
            for (u, v) in list(missing):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    present.add((u, v))
                    missing.discard((u, v))
                    improved = True
                    break
            if improved:
                continue

            if not present or not missing:
                break
            present_list = list(present)
            missing_list = list(missing)

            # 1-swap
            found_swap = False
            for _ in range(300):
                if time.time() - search_start > time_budget:
                    break
                a, b = rng.choice(present_list)
                c, d = rng.choice(missing_list)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                if is_k23_free(adj, c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    present.discard((a, b)); present.add((c, d))
                    missing.add((a, b)); missing.discard((c, d))
                    improved = True
                    found_swap = True
                    break
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            if found_swap:
                continue

            # 2-swap
            found_2swap = False
            if len(present) < 2 or len(missing) < 2:
                break
            present_list = list(present)
            missing_list = list(missing)
            for _ in range(150):
                if time.time() - search_start > time_budget:
                    break
                (a, b), (c, d) = rng.sample(present_list, 2)
                (e, f), (g, h) = rng.sample(missing_list, 2)
                if len({(a, b), (c, d), (e, f), (g, h)}) < 4:
                    continue
                adj[a] &= ~(1 << b); adj[b] &= ~(1 << a)
                adj[c] &= ~(1 << d); adj[d] &= ~(1 << c)
                if is_k23_free(adj, e, f):
                    adj[e] |= (1 << f); adj[f] |= (1 << e)
                    if is_k23_free(adj, g, h):
                        adj[g] |= (1 << h); adj[h] |= (1 << g)
                        present.discard((a, b)); present.discard((c, d))
                        present.add((e, f)); present.add((g, h))
                        missing.add((a, b)); missing.add((c, d))
                        missing.discard((e, f)); missing.discard((g, h))
                        improved = True
                        found_2swap = True
                        break
                    else:
                        adj[e] &= ~(1 << f); adj[f] &= ~(1 << e)
                adj[a] |= (1 << b); adj[b] |= (1 << a)
                adj[c] |= (1 << d); adj[d] |= (1 << c)

            if not found_2swap:
                break

        return adj

    def structured_initial():
        adj = [0] * n
        # Circulant base: each vertex connects to ±1, ±2, ±5 mod 20
        for i in range(n):
            for d in (1, 2, 5):
                for j in ((i + d) % n, (i - d) % n):
                    if j != i and not ((adj[i] >> j) & 1):
                        if is_k23_free(adj, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
        # Greedy fill
        for (u, v) in [(u, v) for u in range(n) for v in range(u + 1, n)]:
            if not ((adj[u] >> v) & 1):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def random_structured_initial(seed=None):
        rng = random.Random(seed)
        adj = [0] * n
        for i in range(n):
            candidates = [j for j in range(n) if j != i and not ((adj[i] >> j) & 1)]
            rng.shuffle(candidates)
            for j in candidates[:4]:
                if is_k23_free(adj, i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for (u, v) in all_edges:
            if not ((adj[u] >> v) & 1):
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    best_adj = None
    best_count = 0

    def consider(adj):
        nonlocal best_adj, best_count
        cnt = edge_count(adj)
        if cnt > best_count:
            best_adj = list(adj)
            best_count = cnt

    inits = [structured_initial()]
    for s in range(3):
        inits.append(random_structured_initial(s))
    for s in range(3):
        inits.append(greedy_build(s))

    for adj in inits:
        consider(adj)
        if time.time() < deadline - 8.0:
            adj_ls = local_search(adj, time_budget=4.0, rng=random.Random(1000))
            consider(adj_ls)

    restart = 0
    while time.time() < deadline - 8.0:
        adj = greedy_build(restart)
        consider(adj)
        if time.time() < deadline - 6.0:
            adj_ls = local_search(adj, time_budget=4.0, rng=random.Random(restart + 5000))
            consider(adj_ls)
        restart += 1

    if best_adj is None:
        best_adj = greedy_build()

    # Final high-precision polish: tight add-only pass on the best graph
    if time.time() < deadline - 3.0:
        final = list(best_adj)
        for (u, v) in [(u, v) for u in range(n) for v in range(u + 1, n)]:
            if not ((final[u] >> v) & 1) and is_k23_free(final, u, v):
                final[u] |= (1 << v)
                final[v] |= (1 << u)
        consider(final)

    return adj_to_edges(best_adj)
