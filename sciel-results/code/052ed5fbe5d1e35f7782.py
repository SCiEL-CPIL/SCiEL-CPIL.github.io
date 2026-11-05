"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random

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

    def induced_matching_coloring_mis(edges, n, max_degree):
        """
        Color edges using a Maximum Independent Set (MIS) heuristic on the conflict graph.
        Each color class is an independent set in the conflict graph, which corresponds
        to an induced matching in the original graph.
        """
        num_edges = len(edges)
        if num_edges == 0:
            return []

        # Build adjacency for original graph
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict neighborhoods
        # Two edges e1=(u1,v1) and e2=(u2,v2) conflict if:
        # 1. They share a vertex.
        # 2. They are joined by an edge (u1-u2, u1-v2, v1-u2, v1-v2).
        # We can define a "forbidden set" of vertices for each edge.
        # If e2 has any vertex in forbidden_set(e1), they conflict.
        # forbidden_set(e) = {u, v} U N(u) U N(v)
        
        forbidden_sets = []
        for u, v in edges:
            fs = set()
            fs.add(u)
            fs.add(v)
            fs.update(adj[u])
            fs.update(adj[v])
            forbidden_sets.append(fs)

        # Build the conflict graph adjacency list
        # conflict_adj[i] is the set of indices j such that edge i conflicts with edge j
        conflict_adj = [set() for _ in range(num_edges)]
        for i in range(num_edges):
            for j in range(i + 1, num_edges):
                # Check conflict
                if forbidden_sets[i] & forbidden_sets[j]:
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)

        color_of_edge = [-1] * num_edges
        uncolored = set(range(num_edges))
        color_id = 0

        while uncolored:
            # Find a large independent set in the subgraph induced by 'uncolored'
            # Heuristic: Greedy Min-Degree
            # 1. Compute degrees in the remaining graph
            # 2. Pick vertex with min degree
            # 3. Add to independent set
            # 4. Remove vertex and its neighbors from 'uncolored' candidates for this color
            
            # To speed up, we can just iterate through uncolored and pick the one with fewest conflicts
            # among those still uncolored.
            
            # Efficiently finding min-degree in induced subgraph:
            # We can maintain a heap or just scan. Since N is small (<= 120 edges approx), scanning is fine.
            
            current_candidates = set(uncolored)
            independent_set = []
            
            while current_candidates:
                # Find vertex with minimum degree in the induced subgraph of current_candidates
                min_deg = float('inf')
                min_vertex = -1
                for v in current_candidates:
                    # Degree in current_candidates
                    deg = len(conflict_adj[v] & current_candidates)
                    if deg < min_deg:
                        min_deg = deg
                        min_vertex = v
                
                if min_vertex == -1:
                    break
                    
                independent_set.append(min_vertex)
                # Remove min_vertex and its neighbors from candidates
                current_candidates.remove(min_vertex)
                neighbors = conflict_adj[min_vertex] & current_candidates
                for nb in neighbors:
                    current_candidates.remove(nb)
            
            # Assign color to all edges in independent_set
            for v in independent_set:
                color_of_edge[v] = color_id
                uncolored.remove(v)
            
            color_id += 1
            
        return color_of_edge

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = induced_matching_coloring_mis(edges, n, d)
        out.append(colors)
    
    return out
# EVOLVE-BLOCK-END
