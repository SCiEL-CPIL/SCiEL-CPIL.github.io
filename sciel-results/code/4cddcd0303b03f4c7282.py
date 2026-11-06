def construct():
    import random

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
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

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        adj_conf = [0] * m
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 in (u2, v2) or v1 in (u2, v2):
                    adj_conf[i] |= (1 << j)
                    adj_conf[j] |= (1 << i)
                else:
                    if u2 in adj_orig[u1] or v2 in adj_orig[u1] or u2 in adj_orig[v1] or v2 in adj_orig[v1]:
                        adj_conf[i] |= (1 << j)
                        adj_conf[j] |= (1 << i)

        def max_is(candidates):
            if candidates == 0:
                return 0, 0

            def greedy(cand):
                size = 0
                mask = 0
                while cand:
                    best_v = -1
                    best_deg = 10**9
                    t = cand
                    while t:
                        b = t & -t
                        v = b.bit_length() - 1
                        deg = (adj_conf[v] & cand).bit_count()
                        if deg < best_deg:
                            best_deg = deg
                            best_v = v
                        t &= t - 1
                    if best_v < 0:
                        break
                    mask |= (1 << best_v)
                    size += 1
                    cand &= ~(adj_conf[best_v] | (1 << best_v))
                return size, mask

            best_size, best_mask = greedy(candidates)

            def bb(cand, cmask, csize):
                nonlocal best_size, best_mask
                if cand == 0:
                    if csize > best_size:
                        best_size = csize
                        best_mask = cmask
                    return
                if csize + cand.bit_count() <= best_size:
                    return
                gs, _ = greedy(cand)
                if csize + gs <= best_size:
                    return
                t = cand
                pivot = -1
                md = -1
                while t:
                    b = t & -t
                    v = b.bit_length() - 1
                    deg = (adj_conf[v] & cand).bit_count()
                    if deg > md:
                        md = deg
                        pivot = v
                    t &= t - 1
                if pivot < 0:
                    return
                bb(cand & ~(adj_conf[pivot] | (1 << pivot)), cmask | (1 << pivot), csize + 1)
                bb(cand & ~(1 << pivot), cmask, csize)

            bb(candidates, 0, 0)
            return best_size, best_mask

        colors = [-1] * m
        avail = (1 << m) - 1
        ci = 0
        while avail:
            _, mm = max_is(avail)
            t = mm
            while t:
                b = t & -t
                idx = b.bit_length() - 1
                colors[idx] = ci
                avail &= ~b
                t &= t - 1
            ci += 1
        return colors

    return [solve_instance(*spec) for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103))]
