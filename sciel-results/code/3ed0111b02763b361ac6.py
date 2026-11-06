import time
import sys
import math
from random import Random

def construct():
    def instance(n, d, seed):
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

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        neigh_set = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh_set[i].add(j)
                    neigh_set[j].add(i)
                else:
                    if (c in adj_orig[a] or dd in adj_orig[a] or
                            c in adj_orig[b] or dd in adj_orig[b]):
                        neigh_set[i].add(j)
                        neigh_set[j].add(i)

        neigh = [list(s) for s in neigh_set]
        deg = [len(neigh[i]) for i in range(m)]

        # Bitset representation of conflict graph for fast operations
        conflict_bits = [0] * m
        for i in range(m):
            bits = 0
            for j in neigh[i]:
                bits |= (1 << j)
            conflict_bits[i] = bits

        def count_conflicts(col):
            cnt = 0
            for i in range(m):
                ci = col[i]
                for j in neigh[i]:
                    if j > i and col[j] == ci:
                        cnt += 1
            return cnt

        def local_search_reduce(col, K, time_limit):
            if time.time() > time_limit:
                return col, False
            for i in range(m):
                if col[i] >= K:
                    col[i] = rng.randrange(K)
            conflicts = count_conflicts(col)
            if conflicts == 0:
                return col, True
            best_col_ls = col[:]
            best_conflicts = conflicts
            T = 1.0
            cooling = 0.995
            max_iters = 300000
            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return col, True
                v = rng.randrange(m)
                old_c = col[v]
                new_c = rng.randrange(K)
                if new_c == old_c:
                    continue
                delta = 0
                for u in neigh[v]:
                    if col[u] == old_c:
                        delta -= 1
                    if col[u] == new_c:
                        delta += 1
                col[v] = new_c
                conflicts += delta
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_col_ls = col[:]
                elif delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
                    pass
                else:
                    col[v] = old_c
                    conflicts -= delta
                T *= cooling
                # Occasionally try a second move on a neighbor to escape local optima
                if it % 500 == 0 and conflicts > 0:
                    v2 = rng.randrange(m)
                    if v2 != v:
                        old_c2 = col[v2]
                        new_c2 = rng.randrange(K)
                        if new_c2 != old_c2:
                            delta2 = 0
                            for u in neigh[v2]:
                                if col[u] == old_c2:
                                    delta2 -= 1
                                if col[u] == new_c2:
                                    delta2 += 1
                            if delta2 < 0 or (delta2 == 0 and rng.random() < 0.5):
                                col[v2] = new_c2
                                conflicts += delta2
                                if conflicts < best_conflicts:
                                    best_conflicts = conflicts
                                    best_col_ls = col[:]
            if best_conflicts == 0:
                return best_col_ls, True
            return col, False

        # Maximum clique of the complement of the conflict graph = size of a
        # maximum induced matching = lower bound on the number of colours.
        def max_clique_bound(deadline):
            comp = [set() for _ in range(m)]
            for i in range(m):
                s = comp[i]
                for j in range(m):
                    if j != i and j not in neigh_set[i]:
                        s.add(j)
            order = sorted(range(m), key=lambda x: len(comp[x]), reverse=True)
            best = [0]
            nodes = [order[i] for i in range(m)]
            idx = [order.index(v) for v in nodes]
            adjb = [comp[v] for v in nodes]
            limit = [0] * m
            for i in range(m - 1, -1, -1):
                limit[i] = (len(nodes) - i) + (best[0] - 1)
            count = [0]

            def bk(R, P, X):
                if count[0] > 2000000:
                    return
                if not P and not X:
                    if len(R) > best[0]:
                        best[0] = len(R)
                    return
                if len(R) + len(P) <= best[0]:
                    return
                if time.time() > deadline:
                    return
                cand = [v for v in P if len(R) + limit[idx[v]] > best[0]]
                if not cand:
                    return
                while cand:
                    count[0] += 1
                    v = cand.pop()
                    Pn = P & adjb[v]
                    Xn = X & adjb[v]
                    P = P - {v}
                    X = X | {v}
                    bk(R | {v}, Pn, Xn)
                    X = X - {v}
            bk(frozenset(), frozenset(nodes), frozenset())
            return best[0]

        # Phase 1: Greedy MIS extraction for a strong initial colouring.
        active = set(range(m))
        color = [-1] * m
        current_color_id = 0
        rng = Random(seed * 12345 + 678)
        while active:
            if time.time() > deadline - 10.0:
                break
            best_set = set()
            for _ in range(5):
                cand_set = set()
                remaining = set(active)
                while remaining:
                    min_v = -1
                    min_deg = float('inf')
                    vertices = list(remaining)
                    rng.shuffle(vertices)
                    for v in vertices:
                        d_v = sum(1 for u in neigh[v] if u in remaining)
                        if d_v < min_deg:
                            min_deg = d_v
                            min_v = v
                    cand_set.add(min_v)
                    remaining.remove(min_v)
                    for u in neigh[min_v]:
                        if u in remaining:
                            remaining.remove(u)
                if len(cand_set) > len(best_set):
                    best_set = cand_set
            for v in best_set:
                color[v] = current_color_id
            active -= best_set
            current_color_id += 1

        if active:
            for v in sorted(active, key=lambda x: -deg[x]):
                used = set()
                for u in neigh[v]:
                    if color[u] != -1:
                        used.add(color[u])
                c = 0
                while c in used:
                    c += 1
                color[v] = c

        current_K = max(color) + 1 if color else 0
        best_color = color[:]

        # Phase 2: Simulated annealing to reduce the colour count.
        lb = 0
        if time.time() < deadline - 8.0:
            try:
                lb = max_clique_bound(deadline - 8.0)
            except Exception:
                lb = 0
        lb = max(2, lb)

        for target_K in range(current_K - 1, lb - 1, -1):
            if time.time() > deadline - 5.0:
                break
            success = False
            for attempt in range(15):
                if time.time() > deadline - 5.0:
                    break
                trial = best_color[:]
                for i in range(m):
                    if rng.random() < 0.4:
                        trial[i] = rng.randrange(target_K)
                for i in range(m):
                    if trial[i] >= target_K:
                        trial[i] = rng.randrange(target_K)
                result, ok = local_search_reduce(trial, target_K, deadline - 5.0)
                if ok:
                    best_color = result
                    current_K = target_K
                    success = True
                    break
            if not success:
                break

        # Final polish: try to reduce further with more aggressive search
        if time.time() < deadline - 2.0 and current_K > lb:
            for target_K in range(current_K - 1, lb - 1, -1):
                if time.time() > deadline - 2.0:
                    break
                for attempt in range(5):
                    if time.time() > deadline - 2.0:
                        break
                    trial = best_color[:]
                    for i in range(m):
                        if rng.random() < 0.5:
                            trial[i] = rng.randrange(target_K)
                    for i in range(m):
                        if trial[i] >= target_K:
                            trial[i] = rng.randrange(target_K)
                    result, ok = local_search_reduce(trial, target_K, deadline - 2.0)
                    if ok:
                        best_color = result
                        current_K = target_K
                        break

        return compress(best_color)

    try:
        start = time.time()
        deadline = start + 145.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = remaining / (len(specs) - idx)
            share = min(share, 50.0)
            share = max(share, 1.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
