"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq

def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, leave 5s safety margin for 150s limit

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

    def color_graph(edges, max_time):
        if not edges:
            return []
        
        # Build adjacency list for the original graph
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        m = len(edges)
        
        # Precompute near vertices for each edge
        near_vertices = []
        for e in edges:
            u, v = e
            near = set([u, v])
            near.update(adj.get(u, set()))
            near.update(adj.get(v, set()))
            near_vertices.append(near)
        
        # Determine conflicts
        conflicts = [set() for _ in range(m)]
        for i in range(m):
            near_i = near_vertices[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                if uj in near_i or vj in near_i:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        
        # Convert to lists for faster iteration
        conflict_lists = [list(s) for s in conflicts]
        
        # Greedy coloring with multiple orderings, keep best
        def greedy_color(order):
            color = [0] * m
            color_count = [0] * m  # count of edges per color
            max_c = 0
            for i in order:
                taken = set()
                for j in conflict_lists[i]:
                    if color[j] != -1:
                        taken.add(color[j])
                c = 0
                while c in taken:
                    c += 1
                color[i] = c
                if c > max_c:
                    max_c = c
            return color, max_c + 1
        
        # Try multiple orderings
        best_colors = None
        best_num_colors = m + 1
        
        # Ordering 1: by conflict degree descending
        order1 = sorted(range(m), key=lambda i: len(conflict_lists[i]), reverse=True)
        c1, n1 = greedy_color(order1)
        if n1 < best_num_colors:
            best_num_colors = n1
            best_colors = c1
        
        # Ordering 2: random
        for _ in range(5):
            if time.time() > max_time:
                break
            order2 = list(range(m))
            random.shuffle(order2)
            c2, n2 = greedy_color(order2)
            if n2 < best_num_colors:
                best_num_colors = n2
                best_colors = c2
        
        # Ordering 3: DSATUR-like (most saturated first, then most constrained)
        # We'll do a simple version: process in order of increasing "slack"
        # Actually, let's try a different heuristic: color by increasing number of available colors
        # This is complex, so let's just try a few more random orders
        
        # Local search: try to reduce number of colors
        # We'll do a simple improvement: for each color class, try to merge
        # This is complex, so let's skip for now
        
        return best_colors

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    for spec in specs:
        if time.time() > DEADLINE:
            # Fallback: simple greedy
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
            continue
        
        edges = instance(*spec)
        # Allocate time: 40s per graph
        time_limit = min(DEADLINE, time.time() + 40)
        colors = color_graph(edges, time_limit)
        out.append(colors)
        
    return out
# EVOLVE-BLOCK-END
