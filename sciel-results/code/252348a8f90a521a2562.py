import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin below 120s

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_incompatible(edges, adj):
        m = len(edges)
        incompatible = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
                    continue
                if (u2 in adj[u1] or v2 in adj[u1] or
                        u2 in adj[v1] or v2 in adj[v1]):
                    incompatible[i].add(j)
                    incompatible[j].add(i)
        return incompatible

    def count_colors(colors):
        if not colors:
            return 0
        return max(colors) + 1

    def dsatur_coloring(incompatible, m, order=None):
        """DSATUR with optional custom vertex ordering for tie-breaking."""
        colors = [-1] * m
        saturation = [set() for _ in range(m)]
        degree = [len(incompatible[i]) for i in range(m)]
        colored = [False] * m
        
        if order is None:
            order = list(range(m))
        order_idx = {v: i for i, v in enumerate(order)}
        
        for _ in range(m):
            best_v = -1
            best_sat = -1
            best_deg = -1
            best_ord = 10**9
            for v in range(m):
                if not colored[v]:
                    sat = len(saturation[v])
                    if (sat > best_sat or
                            (sat == best_sat and degree[v] > best_deg) or
                            (sat == best_sat and degree[v] == best_deg and order_idx[v] < best_ord)):
                        best_v = v
                        best_sat = sat
                        best_deg = degree[v]
                        best_ord = order_idx[v]
            
            if best_v == -1:
                break
            
            used = set(saturation[best_v])
            c = 0
            while c in used:
                c += 1
            colors[best_v] = c
            colored[best_v] = True
            for u in incompatible[best_v]:
                if not colored[u]:
                    saturation[u].add(c)
        
        return colors

    def local_search(incompatible, colors, m, time_limit, rng):
        """Move + swap local search with perturbation to escape local minima."""
        best_colors = colors[:]
        best_k = count_colors(best_colors)
        start = time.time()
        
        def neighbor_conflict(v, c, cols):
            for u in incompatible[v]:
                if cols[u] == c:
                    return True
            return False
        
        while time.time() - start < time_limit:
            improved = False
            
            # 1) Try moving a vertex to a lower color
            for v in rng.sample(range(m), m):
                cv = best_colors[v]
                if cv == 0:
                    continue
                for c in range(cv):
                    if neighbor_conflict(v, c, best_colors):
                        continue
                    best_colors[v] = c
                    improved = True
                    break
                if improved:
                    break
            
            if improved:
                new_k = count_colors(best_colors)
                if new_k < best_k:
                    best_k = new_k
                continue
            
            # 2) Try swapping two vertices' colors
            swapped = False
            for _ in range(40):
                v1, v2 = rng.sample(range(m), 2)
                c1, c2 = best_colors[v1], best_colors[v2]
                if c1 == c2:
                    continue
                ok1 = True
                for u in incompatible[v1]:
                    if u == v2:
                        continue
                    if best_colors[u] == c2:
                        ok1 = False
                        break
                if not ok1:
                    continue
                ok2 = True
                for u in incompatible[v2]:
                    if u == v1:
                        continue
                    if best_colors[u] == c1:
                        ok2 = False
                        break
                if not ok2:
                    continue
                best_colors[v1] = c2
                best_colors[v2] = c1
                swapped = True
                break
            
            if swapped:
                new_k = count_colors(best_colors)
                if new_k < best_k:
                    best_k = new_k
            
            if not improved and not swapped:
                # Stuck: perturb to escape local minimum
                if rng.random() < 0.4:
                    v = rng.randrange(m)
                    new_c = rng.randint(0, best_k)
                    best_colors[v] = new_c
                else:
                    break
        
        return best_colors

    def solve_instance(edges, adj, time_budget):
        m = len(edges)
        if m == 0:
            return []
        
        incompatible = build_incompatible(edges, adj)
        degree = [len(incompatible[i]) for i in range(m)]
        
        best_colors = None
        best_k = 10**9
        rng = random.Random(7)
        
        # Multi-restart DSATUR with diverse orderings
        n_restarts = 8
        restart_budget = time_budget * 0.35
        for r in range(n_restarts):
            if time.time() > start_time + time_budget - 1.0:
                break
            
            if r == 0:
                order = list(range(m))
            elif r == 1:
                order = list(range(m))
                random.Random(100 + r).shuffle(order)
            elif r == 2:
                # Order by descending degree
                order = sorted(range(m), key=lambda v: -degree[v])
            elif r == 3:
                # Order by ascending degree
                order = sorted(range(m), key=lambda v: degree[v])
            else:
                # Random tie-break with degree
                seed_val = 200 + r
                order = sorted(range(m), key=lambda v: (-degree[v], random.Random(seed_val).random()))
            
            colors = dsatur_coloring(incompatible, m, order)
            k = count_colors(colors)
            if k < best_k:
                best_k = k
                best_colors = colors[:]
        
        if best_colors is None:
            best_colors = dsatur_coloring(incompatible, m, list(range(m)))
            best_k = count_colors(best_colors)
        
        # Local search with periodic restarts
        ls_budget = time_budget * 0.65
        cur = best_colors[:]
        cur_k = best_k
        
        while time.time() < start_time + time_budget - 0.5:
            # Run local search for a short burst
            burst_time = min(3.0, max(0.5, time_budget * 0.1))
            cur = local_search(incompatible, cur, m, burst_time, rng)
            new_k = count_colors(cur)
            
            if new_k < best_k:
                best_k = new_k
                best_colors = cur[:]
            else:
                # No improvement: perturb and continue
                if rng.random() < 0.6:
                    v = rng.randrange(m)
                    cur[v] = rng.randint(0, best_k)
                else:
                    break
        
        # Compact color numbering
        remap = {}
        nxt = 0
        out = [0] * m
        for i in range(m):
            c = best_colors[i]
            if c not in remap:
                remap[c] = nxt
                nxt += 1
            out[i] = remap[c]
        
        return out

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = deadline - time.time()
        if remaining <= 2:
            # Fallback: quick greedy
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            incompatible = build_incompatible(edges, adj)
            colors = dsatur_coloring(incompatible, len(edges))
            out.append(colors)
            continue
        
        # Allocate time: first instance gets more, later ones less
        if len(out) == 0:
            time_budget = remaining * 0.45
        elif len(out) == 1:
            time_budget = remaining * 0.35
        else:
            time_budget = remaining * 0.30
        
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colors = solve_instance(edges, adj, time_budget)
        out.append(colors)
    
    return out
