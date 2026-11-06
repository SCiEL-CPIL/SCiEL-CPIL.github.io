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
        # Use a set for faster conflict checking if needed, but list is fine for iteration
        # We need to know if two edges conflict.
        # Precompute conflict matrix? m is small (<= 120). 120x120 = 14400 booleans. Fine.
        # But list of lists is sufficient.
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
        K0 = max_deg + 1
        color0 = [-1] * m
        uncolored = set(range(m))
        
        for _ in range(m):
            # Find vertex with maximum saturation degree
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
            # Assign smallest available color
            used = set()
            for u in neigh[best_v]:
                if color0[u] != -1:
                    used.add(color0[u])
            c = 0
            while c in used:
                c += 1
            color0[best_v] = c
        
        # Local search: try to reduce colour count
        best_color = color0[:]
        best_K = max(c + 1 for c in best_color) if best_color else 0

        rng = Random(seed * 7919 + 42)

        def count_conflicts(color, K):
            cnt = 0
            for i in range(m):
                for j in neigh[i]:
                    if j > i and color[i] == color[j]:
                        cnt += 1
            return cnt

        def local_search_reduce(color, K, time_limit):
            """
            Try to find a valid K-colouring starting from an initial assignment
            using simulated annealing with 2-swaps.
            """
            if time.time() > time_limit:
                return color, False

            # Ensure colour values are in [0, K)
            for i in range(m):
                if color[i] >= K:
                    color[i] = rng.randrange(K)

            conflicts = count_conflicts(color, K)
            if conflicts == 0:
                return color, True

            best_color_ls = color[:]
            best_conflicts = conflicts

            T = 1.0
            T_min = 0.01
            cooling = 0.997
            max_iters = 500000 # Increased iterations for more search
            
            # Precompute neighbor sets for faster delta calc? 
            # neigh is list of lists. 
            # For delta calc of single vertex v:
            # delta = (# neighbors with new_c) - (# neighbors with old_c)
            # For swap u, v:
            # Need to check neighbors of u and v.
            
            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return color, True

                # 10% chance to do a 2-swap, 90% single move
                if rng.random() < 0.1 and m > 1:
                    # Pick two distinct vertices
                    u = rng.randrange(m)
                    v = rng.randrange(m)
                    if u == v:
                        continue
                    
                    cu = color[u]
                    cv = color[v]
                    if cu == cv:
                        continue # Swapping same color does nothing
                    
                    # Calculate delta for swapping u and v
                    # Conflict change:
                    # 1. u loses neighbors with cu, gains neighbors with cv
                    # 2. v loses neighbors with cv, gains neighbors with cu
                    # 3. If u and v are neighbors, they conflict if cu==cv (no, they are different). 
                    #    After swap, u has cv, v has cu. They still don't conflict with each other 
                    #    unless cv==cu which is false. 
                    #    Wait, if u and v are neighbors, they conflict if color[u]==color[v].
                    #    Before: cu != cv -> No conflict between u,v.
                    #    After: cv != cu -> No conflict between u,v.
                    #    So the edge (u,v) doesn't contribute to delta.
                    
                    delta = 0
                    # Neighbors of u
                    for w in neigh[u]:
                        if w == v: continue
                        if color[w] == cu: delta -= 1
                        if color[w] == cv: delta += 1
                        
                    # Neighbors of v
                    for w in neigh[v]:
                        if w == u: continue
                        if color[w] == cv: delta -= 1
                        if color[w] == cu: delta += 1

                    # Apply swap
                    color[u] = cv
                    color[v] = cu
                    conflicts += delta

                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_color_ls = color[:]
                    elif delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
                        pass # Accept
                    else:
                        # Revert
                        color[u] = cu
                        color[v] = cv
                        conflicts -= delta

                else:
                    # Single vertex move
                    # Pick a vertex, preferably one in conflict
                    # Simple: pick random
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
                    elif delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
                        pass
                    else:
                        color[v] = old_c
                        conflicts -= delta

                T *= cooling

            if best_conflicts == 0:
                return best_color_ls, True
            return color, False

        # Try to reduce K from current_K down to lower bound
        current_color = best_color[:]
        current_K = best_K

        for target_K in range(current_K - 1, 2, -1):
            if time.time() > deadline - 5.0:
                break

            success = False
            # Increase attempts for lower K values?
            # Let's try 20 attempts per K value
            for attempt in range(20):
                if time.time() > deadline - 5.0:
                    break
                # Start from current best colouring, randomise some colours
                trial = current_color[:]
                # Randomization strategy: 
                # If target_K is close to current_K, randomize less.
                # If target_K is much lower, randomize more.
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
                # Try a few more with different initialisations (full random)
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
                break  # Can't reduce further

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
