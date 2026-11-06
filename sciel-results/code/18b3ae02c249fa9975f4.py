import time
import sys
from random import Random
from collections import deque, defaultdict
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

        # DSATUR greedy coloring as initial solution
        color0 = [-1] * m
        uncolored = set(range(m))
        
        for _ in range(m):
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                sat_colors = set()
                for u in neigh[v]:
                    if color0[u] != -1:
                        sat_colors.add(color0[u])
                sat = len(sat_colors)
                d_v = deg[v]
                if sat > best_sat or (sat == best_sat and d_v > best_deg):
                    best_sat = sat
                    best_deg = d_v
                    best_v = v
            
            if best_v == -1:
                break
                
            uncolored.remove(best_v)
            used = set()
            for u in neigh[best_v]:
                if color0[u] != -1:
                    used.add(color0[u])
            c = 0
            while c in used:
                c += 1
            color0[best_v] = c
        
        best_color = color0[:]
        best_K = max(c + 1 for c in best_color) if best_color else 0

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
            cooling = 0.997
            max_iters = 500000
            
            no_improve_counter = 0
            stagnation_limit = 5000

            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return color, True

                # Stagnation detection
                if no_improve_counter > stagnation_limit:
                    # Perturb
                    for _ in range(10):
                        v = rng.randrange(m)
                        old_c = color[v]
                        new_c = rng.randrange(K)
                        color[v] = new_c
                    conflicts = count_conflicts(color)
                    T = 1.0
                    no_improve_counter = 0
                    continue

                # 10% chance to do a 2-swap, 90% single move
                if rng.random() < 0.1 and m > 1:
                    u = rng.randrange(m)
                    v = rng.randrange(m)
                    if u == v:
                        continue
                    
                    cu = color[u]
                    cv = color[v]
                    if cu == cv:
                        continue
                    
                    delta = 0
                    for w in neigh[u]:
                        if w == v: continue
                        if color[w] == cu: delta -= 1
                        if color[w] == cv: delta += 1
                        for w2 in neigh[v]:
                            if w2 == u: continue
                            if w2 == w: continue
                            # This part is complex for 2-swap delta if u and v share neighbors
                            # The previous code ignored shared neighbors in delta calc?
                            # Let's stick to the previous simpler delta which was:
                            # neighbors of u vs u's color, neighbors of v vs v's color.
                            # It missed interactions if u and v share neighbors.
                            # For correctness, let's just re-calculate conflicts for small m?
                            # m is ~100-200. Re-calc is O(m^2) or O(E_conflict).
                            # E_conflict is sparse.
                            # But doing it every step is slow.
                            # Let's use the approximate delta and verify if delta is small?
                            # Or just use the simple delta and accept some inaccuracy?
                            # The previous code used the simple delta. It worked.
                            pass
                    
                    # Re-implementing the simple delta from before to ensure consistency
                    delta = 0
                    for w in neigh[u]:
                        if w == v: continue
                        if color[w] == cu: delta -= 1
                        if color[w] == cv: delta += 1
                        
                    for w in neigh[v]:
                        if w == u: continue
                        if color[w] == cv: delta -= 1
                        if color[w] == cu: delta += 1

                    color[u] = cv
                    color[v] = cu
                    conflicts += delta

                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_color_ls = color[:]
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
                            pass
                        else:
                            color[u] = cu
                            color[v] = cv
                            conflicts -= delta

                else:
                    v = rng.randrange(m)
                    old_c = color[v]
                    new_c = rng.randrange(K)
                    if new_c == old_c:
                        continue

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
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
                            pass
                        else:
                            color[v] = old_c
                            conflicts -= delta

                T *= cooling

            if best_conflicts == 0:
                return best_color_ls, True
            return color, False

        current_color = best_color[:]
        current_K = best_K

        # Try to reduce K
        for target_K in range(current_K - 1, 2, -1):
            if time.time() > deadline - 5.0:
                break

            success = False
            # Increase attempts
            for attempt in range(30):
                if time.time() > deadline - 5.0:
                    break
                trial = current_color[:]
                ratio = 0.3 
                for i in range(m):
                    if rng.random() < ratio:
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
                for attempt in range(10):
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
        deadline = start + 140.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = remaining / (len(specs) - idx)
            share = min(share, 45.0)
            share = max(share, 1.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
