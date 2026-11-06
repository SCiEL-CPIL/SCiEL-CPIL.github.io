def construct():
    import time
    import random
    from itertools import combinations
    _T0 = time.monotonic()
    _DEADLINE = 140.0

    def _remaining():
        return _DEADLINE - time.monotonic()

    def _instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e not in edges and degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def _greedy_mis(candidates, adj, edge_index):
        order = sorted(candidates, key=lambda e: (len(adj[e[0]]) + len(adj[e[1]]), e))
        chosen, used = [], set()
        for e in order:
            if e in used:
                continue
            if any(n in used for n in adj[e[0]]) or any(n in used for n in adj[e[1]]):
                continue
            chosen.append(e)
            used.add(e)
            for n in adj[e[0]]:
                used.add(n)
            for n in adj[e[1]]:
                used.add(n)
        return chosen

    def _solve_instance(edges):
        m = len(edges)
        if m == 0:
            return []
        adj = {u: set() for u in range(max(edges)[1] + 1)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        edge_index = {e: i for i, e in enumerate(edges)}

        # conflict graph: edges i,j conflict if share a vertex or are joined by an edge
        conflict = [0] * m
        for i in range(m):
            a, b = edges[i]
            mask = 1 << i
            for n in adj[a]:
                if n < a:
                    continue
                if (n, a) in edge_index:
                    mask |= 1 << edge_index[(n, a)]
                elif (a, n) in edge_index:
                    mask |= 1 << edge_index[(a, n)]
            for n in adj[b]:
                if n < b:
                    continue
                if (n, b) in edge_index:
                    mask |= 1 << edge_index[(n, b)]
                elif (b, n) in edge_index:
                    mask |= 1 << edge_index[(b, n)]
            conflict[i] = mask

        # greedy upper bound via maximal packing of disjoint (non-conflicting) edges
        def greedy_ub(remaining):
            count, rem = 0, remaining
            while rem:
                # pick the edge with smallest degree in remaining
                best, best_deg = -1, 1 << 30
                r = rem
                while r:
                    lsb = r & (-r)
                    i = lsb.bit_length() - 1
                    deg = (conflict[i] & rem).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best = i
                        if deg == 1:
                            break
                    r -= lsb
                if best < 0:
                    break
                count += 1
                rem &= ~(conflict[best] | (1 << best))
            return count

        # branch and bound for max independent set on conflict graph
        best_set = [0]
        best_size = [0]
        deadline = _DEADLINE - 1.0

        def bb(remaining, chosen):
            if time.monotonic() > deadline:
                return
            if remaining == 0:
                if len(chosen) > best_size[0]:
                    best_size[0] = len(chosen)
                    best_set[0] = sum(1 << i for i in chosen)
                return
            ub = len(chosen) + greedy_ub(remaining)
            if ub <= best_size[0]:
                return
            # pick a vertex: smallest degree in remaining
            lsb = remaining & (-remaining)
            v = lsb.bit_length() - 1
            neigh = conflict[v] & remaining
            # branch: include v
            bb(remaining & ~neigh & ~(1 << v), chosen + [v])
            # branch: exclude v
            bb(remaining & ~(1 << v), chosen)

        bb((1 << m) - 1, [])

        # build colouring: repeatedly extract a large independent set (induced matching)
        colour = {}
        remaining = (1 << m) - 1
        c = 0
        while remaining:
            if time.monotonic() > deadline:
                break
            # find a large independent set via bb on remaining
            local_best = [0]
            local_size = [0]
            def bb_local(rem, chosen):
                if rem == 0:
                    if len(chosen) > local_size[0]:
                        local_size[0] = len(chosen)
                        local_best[0] = sum(1 << i for i in chosen)
                    return
                ub = len(chosen) + greedy_ub(rem)
                if ub <= local_size[0]:
                    return
                lsb = rem & (-rem)
                v = lsb.bit_length() - 1
                neigh = conflict[v] & rem
                bb_local(rem & ~neigh & ~(1 << v), chosen + [v])
                bb_local(rem & ~(1 << v), chosen)
            bb_local(remaining, [])
            chosen_mask = local_best[0]
            if chosen_mask == 0:
                # fallback: single edge
                lsb = remaining & (-remaining)
                chosen_mask = lsb
            for i in range(m):
                if chosen_mask >> i & 1:
                    colour[i] = c
            remaining &= ~chosen_mask
            c += 1
        # assign any leftover (shouldn't happen) with new colours
        for i in range(m):
            if i not in colour:
                colour[i] = c
                c += 1
        return [colour[i] for i in range(m)]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if _remaining() <= 2.0:
            # not enough time; fall back to a quick greedy for this instance
            edges = _instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj.get(x, set())}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
            continue
        edges = _instance(*spec)
        out.append(_solve_instance(edges))
    return out
