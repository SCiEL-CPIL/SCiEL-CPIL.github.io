import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

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

    def dsatur_coloring(incompatible, m):
        """DSATUR: color vertex with highest saturation first, tie-break by degree."""
        colors = [-1] * m
        saturation = [set() for _ in range(m)]
        degree = [len(incompatible[i]) for i in range(m)]
        colored = [False] * m
        
        for _ in range(m):
            # Pick uncolored vertex with highest saturation, then highest degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(m):
                if not colored[v]:
                    sat = len(saturation[v])
                    if sat > best_sat or (sat == best_sat and degree[v] > best_deg):
                        best_v = v
                        best_sat = sat
                        best_deg = degree[v]
            
            if best_v == -1:
                break
            
            # Assign smallest available color
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

    def count_colors(colors):
        return max(colors) + 1 if colors else 0

    def local_search(incompatible, colors, m, time_limit):
        """Local search: try moves and swaps to reduce color count."""
        best_colors = colors[:]
        best_k = count_colors(best_colors)
        rng = random.Random(42)
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            
            # Try moving a vertex to a lower color
            for v in rng.sample(range(m), m):
                cv = best_colors[v]
                if cv == 0:
                    continue
                # Try each lower color
                for c in range(cv):
                    if c in [best_colors[u] for u in incompatible[v]]:
                        continue
                    # Check if c is used by any neighbor
                    conflict = False
                    for u in incompatible[v]:
                        if best_colors[u] == c:
                            conflict = True
                            break
                    if not conflict:
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
            
            # Try swapping two vertices
            swapped = False
            for _ in range(50):
                v1, v2 = rng.sample(range(m), 2)
                c1, c2 = best_colors[v1], best_colors[v2]
                if c1 == c2:
                    continue
                # Check if swap is valid
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
                # Perform swap
                best_colors[v1] = c2
                best_colors[v2] = c1
                swapped = True
                break
            
            if swapped:
                new_k = count_colors(best_colors)
                if new_k < best_k:
                    best_k = new_k
            
            if not improved and not swapped:
                # No improvement found; try random perturbation occasionally
                if rng.random() < 0.1:
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
        
        # Build incompatibility graph
        incompatible = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
                    continue
                if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
        
        # DSATUR initial coloring
        colors = dsatur_coloring(incompatible, m)
        k = count_colors(colors)
        
        # Local search with time budget
        colors = local_search(incompatible, colors, m, time_budget)
        
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = deadline - time.time()
        if remaining <= 1:
            # Fallback: just use greedy
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
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
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                        incompatible[i].add(j)
                        incompatible[j].add(i)
            colors = dsatur_coloring(incompatible, m)
            out.append(colors)
            continue
        
        # Split remaining time among instances
        time_budget = remaining * 0.3
        if len(out) == 0:
            time_budget = remaining * 0.4
        elif len(out) == 1:
            time_budget = remaining * 0.35
        
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colors = solve_instance(edges, adj, time_budget)
        out.append(colors)
        
    return out
