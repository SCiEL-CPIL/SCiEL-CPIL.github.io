"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from collections import defaultdict
    
    def instance(n, d, seed):
        import random
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
    
    def build_conflict_graph(edges, adj):
        """Build conflict graph: two edges conflict if they share a vertex or are connected by a path of length 2."""
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflict = [set() for _ in range(m)]
        
        for i, e1 in enumerate(edges):
            u1, v1 = e1
            # Edges sharing a vertex with e1
            for x in (u1, v1):
                for y in adj.get(x, set()):
                    for z in adj.get(y, set()):
                        if z == x:
                            continue
                        # Edge (y, z) conflicts with e1 if it doesn't share a vertex with e1
                        # Actually, let's think: e1 = (u1, v1), e2 = (y, z)
                        # They conflict if they share a vertex OR if there's a path of length 2 between them
                        # Path of length 2: u1--y--z where y is adjacent to both u1 and z, but e1 and e2 don't share a vertex
                        # More simply: e1 and e2 conflict if dist(e1, e2) <= 1 in line graph, or dist = 2
                        # In line graph, dist=1 means share vertex, dist=2 means there's an edge connecting an endpoint of e1 to an endpoint of e2
                        pass
        
        # Better approach: for each pair, check if they can be in same induced matching
        # Two edges (a,b) and (c,d) can be in same induced matching iff:
        # 1. No shared vertices: a,b,c,d all distinct
        # 2. No edge between {a,b} and {c,d}: a not adj c, a not adj d, b not adj c, b not adj d
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check shared vertices
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if any endpoint of e1 is adjacent to any endpoint of e2
                if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                    u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        return conflict
    
    def greedy_coloring(conflict, m):
        """Greedy coloring of conflict graph to minimize number of colors."""
        # Order edges by conflict degree (ascending) to help packing
        deg = [len(conflict[i]) for i in range(m)]
        order = sorted(range(m), key=lambda i: deg[i])
        
        color = [-1] * m
        for i in order:
            # Find smallest color not used by neighbors
            used = set()
            for j in conflict[i]:
                if color[j] != -1:
                    used.add(color[j])
            c = 0
            while c in used:
                c += 1
            color[i] = c
        
        # Renumber colors to be 0..k-1
        max_color = max(color) if color else 0
        color_map = {}
        new_color = [0] * m
        for i in range(m):
            if color[i] not in color_map:
                color_map[color[i]] = len(color_map)
            new_color[i] = color_map[color[i]]
        
        return new_color
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        m = len(edges)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build conflict graph
        conflict = build_conflict_graph(edges, adj)
        
        # Greedy coloring
        color = greedy_coloring(conflict, m)
        
        out.append(color)
    
    return out
# EVOLVE-BLOCK-END
