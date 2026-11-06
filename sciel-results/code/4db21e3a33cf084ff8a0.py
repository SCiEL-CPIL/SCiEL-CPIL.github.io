def construct():
    import time
    import random
    from random import Random

    start_time = time.time()
    DEADLINE = 130  # 20s safety margin

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

    def build_conflict_graph(edges, adj):
        m = len(edges)
        # edge_to_idx
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        # For each edge, find all edges that conflict with it
        conflict = [set() for _ in range(m)]
        # For each edge (u,v), conflicting edges are:
        # 1. All edges sharing u or v
        # 2. All edges (x,y) where x in adj[u] and y in adj[v] or x in adj[v] and y in adj[u]
        # Actually: edges joined by an edge means there exists an edge (a,b) in the graph
        #    such that {a,b} connects one endpoint of e1 to one endpoint of e2.
        # More precisely: two edges e1=(u,v) and e2=(w,x) conflict if:
        #   - They share a vertex, OR
        #   - There exists an edge in the graph connecting a vertex of e1 to a vertex of e2
        #     (i.e., w in adj[u] or w in adj[v] or x in adj[u] or x in adj[v])
        
        # Build a map from vertex to list of edge indices
        vertex_edges = [[] for _ in range(max(len(adj), 1))]
        for i, (u, v) in enumerate(edges):
            vertex_edges[u].append(i)
            vertex_edges[v].append(i)
        
        for i, (u, v) in enumerate(edges):
            # All edges sharing u or v
            for j in vertex_edges[u]:
                if j != i:
                    conflict[i].add(j)
            for j in vertex_edges[v]:
                if j != i:
                    conflict[i].add(j)
            # Edges joined by an edge: for each neighbor of u, add all edges incident to that neighbor
            # that don't share a vertex with e (already handled above)
            for w in adj.get(u, ()):
                for j in vertex_edges[w]:
                    conflict[i].add(j)
            for w in adj.get(v, ()):
                for j in vertex_edges[w]:
                    conflict[i].add(j)
        
        # Convert to lists
        conflict = [list(s) for s in conflict]
        return conflict

    def dsatur_coloring(conflict, m, rng):
        colors = [-1] * m
        sat = [set() for _ in range(m)]
        degrees = [len(conflict[i]) for i in range(m)]
        uncolored = set(range(m))
        
        while uncolored:
            # Pick vertex with max saturation, tie-break by degree
            best_v = max(uncolored, key=lambda v: (len(sat[v]), degrees[v]))
            # Find smallest available color
            c = 0
            while c in sat[best_v]:
                c += 1
            colors[best_v] = c
            uncolored.remove(best_v)
            # Update neighbors
            for nb in conflict[best_v]:
                if nb in uncolored:
                    sat[nb].add(c)
        
        return colors

    def local_search(conflict, m, colors, rng, max_iters=5000):
        # Try to reduce number of colors
        # Repeatedly try to move vertices from highest color class to lower colors
        max_c = max(colors) if colors else 0
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            max_c = max(colors)
            # Get vertices in highest color class
            highest_class = [i for i in range(m) if colors[i] == max_c]
            rng.shuffle(highest_class)
            for v in highest_class:
                # Try to move v to a lower color
                forbidden = set()
                for nb in conflict[v]:
                    if colors[nb] != -1:
                        forbidden.add(colors[nb])
                # Find smallest available color < max_c
                for c in range(max_c):
                    if c not in forbidden:
                        colors[v] = c
                        improved = True
                        break
        return colors

    def solve_one(spec):
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        m = len(edges)
        if m == 0:
            return []
        
        conflict = build_conflict_graph(edges, adj)
        
        best_colors = None
        best_num_colors = float('inf')
        
        # Try multiple restarts with different random seeds
        for trial in range(20):
            if time.time() > DEADLINE:
                break
            rng = Random(seed + 1000 * trial)
            colors = dsatur_coloring(conflict, m, rng)
            colors = local_search(conflict, m, colors, rng, max_iters=3000)
            num_colors = max(colors) + 1 if colors else 0
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_colors = colors[:]
        
        if best_colors is None:
            # Fallback: greedy
            best_colors = dsatur_coloring(conflict, m, Random(seed))
        
        return best_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE:
            # Emergency fallback: greedy first-fit
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
        else:
            out.append(solve_one(spec))
    
    return out
