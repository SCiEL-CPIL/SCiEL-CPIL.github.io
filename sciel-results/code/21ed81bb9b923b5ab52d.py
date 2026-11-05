"""Strong edge colourings via maximum induced matching (branch-and-bound on conflict graph)."""
# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    DEADLINE = 130.0

    def instance(n, d, seed):
        from random import Random
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_one(edges):
        m = len(edges)
        if m == 0:
            return []
        # Build adjacency (vertex -> set of vertices)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        # edge index
        eidx = {e: i for i, e in enumerate(edges)}
        # conflict bitsets: two edges conflict if they share a vertex or are joined by an edge
        conf = [0] * m
        for i in range(m):
            u, v = edges[i]
            mask = 1 << i
            # edges sharing u or v
            for nb in adj[u]:
                for e2 in adj.get(nb, ()):
                    pass
            # simpler: iterate all edges (m small)
            for j in range(m):
                if i == j:
                    continue
                a, b = edges[j]
                if a == u or a == v or b == u or b == v:
                    mask |= 1 << j
                    continue
                # joined by an edge: (u-a) or (u-b) or (v-a) or (v-b) in edges
                if (min(u, a), max(u, a)) in eidx or (min(u, b), max(u, b)) in eidx or \
                   (min(v, a), max(v, a)) in eidx or (min(v, b), max(v, b)) in eidx:
                    mask |= 1 << j
            conf[i] = mask

        # greedy initial coloring (DSATUR-like) to get an upper bound
        order = list(range(m))
        # saturation-degree ordering
        col = [-1] * m
        used = {}
        # initial: pick highest degree first
        deg = [bin(conf[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda i: -deg[i])
        for i in order:
            taken = set()
            cm = conf[i]
            for j in range(m):
                if col[j] != -1 and (cm >> j) & 1:
                    taken.add(col[j])
            c = 0
            while c in taken:
                c += 1
            col[i] = c
            used[c] = used.get(c, 0) + 1
        upper = max(col) + 1

        # Iterative: repeatedly extract a maximum induced matching from remaining edges
        remaining = list(range(m))
        assignment = [-1] * m
        color_count = 0

        def max_induced_matching(cand):
            # branch-and-bound to find a large independent set in conflict graph restricted to cand
            best = []
            best_mask = 0
            best_size = 0

            # lower bound via greedy
            gm = 0
            gc = []
            cm = cand
            while cm:
                # pick vertex with min conflicts within remaining
                low = cm & -cm
                li = low.bit_length() - 1
                # choose min-degree in cand
                best_i = li
                best_d = 999
                mm = cm
                while mm:
                    b = mm & -mm
                    bi = b.bit_length() - 1
                    d = bin(conf[bi] & cm).count('1')
                    if d < best_d:
                        best_d = d
                        best_i = bi
                    mm ^= b
                gm |= 1 << best_i
                gc.append(best_i)
                cm &= ~conf[best_i]
            best_size = len(gc)
            best = gc[:]
            best_mask = gm

            # branch and bound (bounded by time)
            nodes = [0]
            def bt(avail, chosen, size):
                nonlocal best, best_size, best_mask
                nodes[0] += 1
                if time.time() > DEADLINE:
                    return
                # upper bound: size + popcount(avail)
                if size + avail.bit_count() <= best_size:
                    return
                if avail == 0:
                    if size > best_size:
                        best_size = size
                        best = chosen[:]
                        best_mask = 0
                        for x in chosen:
                            best_mask |= 1 << x
                    return
                # pick branch vertex: min degree in avail
                mm = avail
                best_i = -1
                best_d = 999
                while mm:
                    b = mm & -mm
                    bi = b.bit_length() - 1
                    d = bin(conf[bi] & avail).count('1')
                    if d < best_d:
                        best_d = d
                        best_i = bi
                    mm ^= b
                # branch 1: include best_i
                chosen.append(best_i)
                bt(avail & ~conf[best_i], chosen, size + 1)
                chosen.pop()
                # branch 2: exclude best_i
                bt(avail & ~(1 << best_i), chosen, size)

            bt(cand, [], 0)
            return best

        while remaining and time.time() < DEADLINE:
            # candidate set = remaining
            best = max_induced_matching(0)
            # rebuild: max_induced_matching expects mask; fix
            # recompute with mask
            def max_induced_matching_mask(cand_mask):
                best = []
                best_size = 0
                # greedy lower bound
                gm = 0
                gc = []
                cm = cand_mask
                while cm:
                    mm = cm
                    best_i = -1
                    best_d = 999
                    while mm:
                        b = mm & -mm
                        bi = b.bit_length() - 1
                        d = bin(conf[bi] & cm).count('1')
                        if d < best_d:
                            best_d = d
                            best_i = bi
                        mm ^= b
                    gm |= 1 << best_i
                    gc.append(best_i)
                    cm &= ~conf[best_i]
                best_size = len(gc)
                best = gc[:]

                nodes = [0]
                def bt(avail, chosen, size):
                    nonlocal best, best_size
                    nodes[0] += 1
                    if time.time() > DEADLINE:
                        return
                    if size + avail.bit_count() <= best_size:
                        return
                    if avail == 0:
                        if size > best_size:
                            best_size = size
                            best = chosen[:]
                        return
                    mm = avail
                    bi_sel = -1
                    best_d = 999
                    while mm:
                        b = mm & -mm
                        bi = b.bit_length() - 1
                        d = bin(conf[bi] & avail).count('1')
                        if d < best_d:
                            best_d = d
                            bi_sel = bi
                        mm ^= b
                    chosen.append(bi_sel)
                    bt(avail & ~conf[bi_sel], chosen, size + 1)
                    chosen.pop()
                    bt(avail & ~(1 << bi_sel), chosen, size)

                bt(cand_mask, [], 0)
                return best

            mask = 0
            for i in remaining:
                mask |= 1 << i
            best = max_induced_matching_mask(mask)
            if not best:
                break
            for i in best:
                assignment[i] = color_count
            color_count += 1
            bestset = set(best)
            remaining = [i for i in remaining if i not in bestset]

        # color any leftovers greedily
        if remaining and time.time() < DEADLINE:
            for i in remaining:
                taken = set()
                cm = conf[i]
                for j in range(m):
                    if assignment[j] != -1 and (cm >> j) & 1:
                        taken.add(assignment[j])
                c = 0
                while c in taken:
                    c += 1
                assignment[i] = c
            color_count = max(assignment) + 1

        return assignment

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 5:
            break
        edges = instance(*spec)
        out.append(solve_one(edges))

    # If we ran out of time, fill missing instances with greedy
    if len(out) < 3:
        # fallback greedy for missing
        existing = len(out)
        specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
        for idx in range(existing, 3):
            edges = instance(*specs[idx])
            m = len(edges)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            eidx = {e: i for i, e in enumerate(edges)}
            col = {}
            for e in edges:
                near = set(e) | {y for x in e for y in adj.get(x, ())}
                taken = {c for f, c in col.items() if set(f) & near}
                col[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([col[e] for e in edges])
    return out
# EVOLVE-BLOCK-END
