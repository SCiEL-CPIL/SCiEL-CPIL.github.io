import time
import sys
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

        # Build adjacency for original graph
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Build conflict graph: two edges conflict if they share a vertex
        # or are joined by an edge (i.e., an endpoint of one is adjacent to an endpoint of the other)
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
        max_deg = max(deg) if deg else 0

        # DSATUR greedy coloring for a good initial coloring
        def dsatur_coloring():
            color = [-1] * m
            uncolored = set(range(m))
            # Track saturation: number of distinct colors in neighbors
            sat = [set() for _ in range(m)]
            # Track degree for tie-breaking
            while uncolored:
                # Pick vertex with max saturation, break ties by degree
                best_v = None
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    s = len(sat[v])
                    if s > best_sat or (s == best_sat and deg[v] > best_deg):
                        best_v = v
                        best_sat = s
                        best_deg = deg[v]
                # Assign smallest available color
                forbidden = set()
                for u in neigh[best_v]:
                    if color[u] != -1:
                        forbidden.add(color[u])
                c = 0
                while c in forbidden:
                    c += 1
                color[best_v] = c
                uncolored.remove(best_v)
                for u in neigh[best_v]:
                    if u in uncolored:
                        sat[u].add(c)
            return color

        color0 = dsatur_coloring()
        K0 = max(color0) + 1

        rng = Random(seed * 7919 + 42)

        def count_conflicts(color):
            cnt = 0
            for i in range(m):
                for j in neigh[i]:
                    if j > i and color[i] == color[j]:
                        cnt += 1
            return cnt

        def local_search_reduce(color, K, time_limit, max_iters=500000):
            """Min-conflicts with simulated annealing to find K-coloring."""
            import math

            # Ensure colours are in [0, K)
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

            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return color, True

                # Pick a random vertex with conflicts, or random vertex
                # Prefer vertices involved in conflicts
                if rng.random() < 0.7:
                    # Pick from vertices that have at least one conflict
                    conflicted = []
                    for v in range(m):
                        for u in neigh[v]:
                            if color[u] == color[v]:
                                conflicted.append(v)
                                break
                    if conflicted:
                        v = conflicted[rng.randrange(len(conflicted))]
                    else:
                        v = rng.randrange(m)
                else:
                    v = rng.randrange(m)

                old_c = color[v]
                new_c = rng.randrange(K)
                if new_c == old_c:
                    continue

                # Compute delta conflicts
                delta = 0
                for u in neigh[v]:
                    if color[u] == old_c:
                        delta -= 1
                    if color[u] == new_c:
                        delta += 1

                color[v] = new_c
                conflicts += delta

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_color_ls = color[:]
                elif delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
                    pass  # Accept
                else:
                    # Reject
                    color[v] = old_c
                    conflicts -= delta

                T *= cooling
                if T < T_min:
                    T = T_min

            if best_conflicts == 0:
                return best_color_ls, True
            return color, False

        # Try to reduce K from K0 down to lower bound
        current_color = color0[:]
        current_K = K0

        # Lower bound: at least the size of the maximum clique in conflict graph
        # For now use a simple bound
        lower_bound = 3

        for target_K in range(current_K - 1, lower_bound - 1, -1):
            if time.time() > deadline - 3.0:
                break

            success = False
            # Try multiple restarts
            for attempt in range(8):
                if time.time() > deadline - 3.0:
                    break
                # Start from current best, perturb
                trial = current_color[:]
                for i in range(m):
                    if trial[i] >= target_K:
                        trial[i] = rng.randrange(target_K)
                    elif rng.random() < 0.2:
                        trial[i] = rng.randrange(target_K)

                result, ok = local_search_reduce(trial, target_K, deadline - 3.0)
                if ok:
                    current_color = result
                    current_K = target_K
                    success = True
                    break

            if not success:
                # Try a few random initializations
                for attempt in range(4):
                    if time.time() > deadline - 3.0:
                        break
                    trial = [rng.randrange(target_K) for _ in range(m)]
                    result, ok = local_search_reduce(trial, target_K, deadline - 3.0)
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
        deadline = start + 140.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            elapsed = time.time() - start
            remaining = deadline - start - elapsed
            share = remaining / (len(specs) - idx)
            share = min(share, 45.0)
            share = max(share, 1.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
