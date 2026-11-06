import time
import sys
from random import Random
import math

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

        # Build conflict graph on edges
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Conflict: two edges conflict if they share a vertex OR are joined by an edge
        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)

        deg = [len(neigh[i]) for i in range(m)]

        # Phase 1: Greedy coloring to get initial upper bound
        color = [-1] * m
        active = set(range(m))
        current_color_id = 0
        rng = Random(seed * 12345 + 678)

        while active:
            if time.time() > deadline - 10.0:
                break

            # Find a large independent set in the active subgraph
            best_set = set()
            for _ in range(8):
                cand_set = set()
                remaining = set(active)
                while remaining:
                    # Pick vertex with min degree in remaining
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

        def count_conflicts(col):
            cnt = 0
            for i in range(m):
                for j in neigh[i]:
                    if j > i and col[i] == col[j]:
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
            cooling = 0.997
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

            if best_conflicts == 0:
                return best_col_ls, True
            return col, False

        # Try to reduce K
        for target_K in range(current_K - 1, 2, -1):
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
