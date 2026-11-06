import time
import sys
import math
from random import Random
from collections import deque

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

        # Build adjacency of original graph
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Build conflict graph on edges: two edges conflict if they share a vertex
        # or are joined by an edge in the original graph
        neigh = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                # Check if they share a vertex
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].add(j)
                    neigh[j].add(i)
                else:
                    # Check if joined by an edge
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].add(j)
                        neigh[j].add(i)

        deg = [len(neigh[i]) for i in range(m)]
        max_deg = max(deg) if deg else 0

        # Greedy coloring to get initial K
        def greedy_coloring():
            color = [-1] * m
            for i in range(m):
                forbidden = set()
                for u in neigh[i]:
                    if color[u] != -1:
                        forbidden.add(color[u])
                c = 0
                while c in forbidden:
                    c += 1
                color[i] = c
            return color, max(color) + 1

        color0, K0 = greedy_coloring()
        current_color = color0[:]
        current_K = K0

        rng = Random(seed * 7919 + 42)

        def count_conflicts(color):
            cnt = 0
            for i in range(m):
                for j in neigh[i]:
                    if j > i and color[i] == color[j]:
                        cnt += 1
            return cnt

        def local_search_reduce(color, K, time_limit):
            if time.time() > time_limit:
                return color, False

            for i in range(m):
                if color[i] >= K:
                    color[i] = rng.randrange(K)

            conflicts = count_conflicts(color)
            if conflicts == 0:
                return color, True

            best_color_ls = color[:]
            best_conflicts = conflicts

            T = 1.0
            T_min = 0.01
            cooling = 0.998
            max_iters = 500000
            stall_count = 0
            stall_limit = 5000

            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return color, True

                v = rng.randrange(m)
                old_c = color[v]

                old_conflicts = 0
                for u in neigh[v]:
                    if color[u] == old_c:
                        old_conflicts += 1

                if old_conflicts == 0:
                    continue

                best_new_c = old_c
                best_new_conflicts = old_conflicts
                for new_c in range(K):
                    if new_c == old_c:
                        continue
                    new_conflicts = 0
                    for u in neigh[v]:
                        if color[u] == new_c:
                            new_conflicts += 1
                    if new_conflicts < best_new_conflicts:
                        best_new_conflicts = new_conflicts
                        best_new_c = new_c

                new_c = best_new_c
                delta = best_new_conflicts - old_conflicts

                color[v] = new_c
                conflicts += delta

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_color_ls = color[:]
                    stall_count = 0
                else:
                    stall_count += 1

                if delta > 0:
                    prob = math.exp(-delta / max(T, 0.001))
                    if rng.random() > prob:
                        color[v] = old_c
                        conflicts -= delta

                T *= cooling

                if stall_count > stall_limit:
                    for i in range(m):
                        if rng.random() < 0.5:
                            color[i] = rng.randrange(K)
                    conflicts = count_conflicts(color)
                    stall_count = 0
                    T = 1.0

            if best_conflicts == 0:
                return best_color_ls, True
            return color, False

        # Try to reduce K
        for target_K in range(current_K - 1, 2, -1):
            if time.time() > deadline - 5.0:
                break

            success = False
            num_restarts = 15
            for attempt in range(num_restarts):
                if time.time() > deadline - 5.0:
                    break

                trial = current_color[:]
                perturbation_rate = 0.3 + 0.3 * (attempt / num_restarts)
                for i in range(m):
                    if rng.random() < perturbation_rate:
                        trial[i] = rng.randrange(target_K)

                for i in range(m):
                    if trial[i] >= target_K:
                        trial[i] = rng.randrange(target_K)

                result, ok = local_search_reduce(trial, target_K, deadline - 5.0)
                if ok:
                    current_color = result
                    current_K = target_K
                    success = True
                    break

            if not success:
                for attempt in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    trial = [rng.randrange(target_K) for _ in range(m)]
                    result, ok = local_search_reduce(trial, target_K, deadline - 5.0)
                    if ok:
                        current_color = result
                        current_K = target_K
                        success = True
                        break

            if not success:
                break

        return compress(current_color)

    try:
        start = time.time()
        deadline = start + 130.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = remaining / (len(specs) - idx)
            share = min(share, 40.0)
            share = max(share, 1.0)
            instance_deadline = start + (time.time() - start) + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
