"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict

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

    def dsatur_coloring(conflict_adj, num_vertices):
        """
        Colors a graph using the DSATUR heuristic.
        conflict_adj: list of sets, where conflict_adj[i] is the set of neighbors of i.
        num_vertices: number of vertices.
        Returns: list of colors for each vertex.
        """
        colors = [-1] * num_vertices
        # sat[i] is the set of colors used by neighbors of i
        sat = [set() for _ in range(num_vertices)]
        
        # Precompute degrees for tie-breaking
        degrees = [len(conflict_adj[i]) for i in range(num_vertices)]
        
        uncolored = set(range(num_vertices))
        
        while uncolored:
            # Find the vertex with the highest saturation degree
            # Tie-break by degree in conflict graph
            best_v = None
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                s = len(sat[v])
                d = degrees[v]
                if s > best_sat or (s == best_sat and d > best_deg):
                    best_sat = s
                    best_deg = d
                    best_v = v
            
            # Assign the smallest available color
            used_colors = sat[best_v]
            c = 0
            while c in used_colors:
                c += 1
            
            colors[best_v] = c
            uncolored.remove(best_v)
            
            # Update saturation sets of neighbors
            for u in conflict_adj[best_v]:
                if u in uncolored:
                    sat[u].add(c)

        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        n = len(edges)
        
        # Build original graph adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build conflict graph
        # conflict[i] contains indices j such that edge i and edge j are in conflict
        conflict_adj = [set() for _ in range(n)]
        
        for i in range(n):
            u1, v1 = edges[i]
            for j in range(i + 1, n):
                u2, v2 = edges[j]
                
                # Check if edges are adjacent (share a vertex)
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)
                else:
                    # Check if edges are distance 2
                    # This happens if there is an edge connecting an endpoint of e_i to an endpoint of e_j
                    # i.e., (u1, u2) in E, (u1, v2) in E, (v1, u2) in E, or (v1, v2) in E
                    # We can check if any of these pairs are in the original edge set.
                    # To make this fast, we can check if the pairs are in adj.
                    # Actually, we just need to know if the pair exists in the graph.
                    # Since we have adj, we can check:
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)

        # Color the conflict graph
        colors = dsatur_coloring(conflict_adj, n)
        
        # The colors are already in the order of edges[0], edges[1], ...
        out.append(colors)

    return out
# EVOLVE-BLOCK-END
