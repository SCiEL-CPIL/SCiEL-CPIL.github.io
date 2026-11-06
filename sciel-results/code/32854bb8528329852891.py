# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        m = len(edges)
        # Build adjacency matrix for Line Graph compatibility
        # cost[i][j] = 0 if edge i can be followed by edge j, else BIG
        # Edge i can be followed by edge j if they share a vertex.
        # We don't need to worry about orientation here because 
        # the bipartite matching on the line graph handles the chaining.
        # If they share a vertex, we can orient them to connect.
        
        BIG = 10**9
        cost = np.full((m, m), BIG, dtype=np.int64)
        
        # Optimization: Precompute vertex sets for each edge
        edge_vertices = [set(e) for e in edges]
        
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                # Check if edges share a vertex
                if edge_vertices[i].intersection(edge_vertices[j]):
                    cost[i][j] = 0
        
        # Find minimum cost perfect matching
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Count valid matches
        valid_matches = 0
        next_edge = [-1] * m
        prev_edge = [-1] * m
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] == 0:
                valid_matches += 1
                next_edge[r] = c
                prev_edge[c] = r
        
        # Build paths
        paths = []
        for i in range(m):
            if prev_edge[i] == -1:
                path_edges = [i]
                cur = i
                while next_edge[cur] != -1:
                    cur = next_edge[cur]
                    path_edges.append(cur)
                
                # Construct vertex list
                # We need to determine the orientation.
                # The chain is e_{p0} -> e_{p1} -> ...
                # We need to orient them consistently.
                # Start with e_{p0}. Pick an arbitrary orientation.
                # Then for the next edge, it must share the end vertex of the previous one.
                
                vertices = []
                # Determine start vertex of the path
                # e_{p0} = (u, v). Let's say we start at u.
                # But we need to ensure the next edge connects to the end.
                
                # Let's trace the vertices.
                # e_0 is the first edge.
                # Let current_vertex be the start of the path.
                # This is tricky without explicit orientation in the matching.
                # However, we can reconstruct the path by walking.
                
                # Let's just store the edge indices and convert later?
                # Or convert now.
                
                # To convert, we need to know the sequence of vertices.
                # e_0 = {a, b}. e_1 = {b, c} (if they share b).
                # If we start at a, go to b, then to c.
                # How to decide start?
                # The path is simple.
                # We can pick either endpoint of the first edge as the start.
                # Then the rest is forced.
                
                # Let's pick edges[path_edges[0]]
                u, v = edges[path_edges[0]]
                # Try starting at u
                curr = u
                next_v = v
                # Check if next edge (if any) connects to next_v
                if len(path_edges) > 1:
                    e_next = edges[path_edges[1]]
                    if next_v not in e_next:
                        # Start must be v
                        curr = v
                        next_v = u
                    else:
                        # Start is u
                        pass
                else:
                    # Single edge, start at u (arbitrary)
                    pass
                
                vertices.append(curr)
                for k in range(len(path_edges)):
                    e_idx = path_edges[k]
                    u_e, v_e = edges[e_idx]
                    # The next vertex in the path is the one that is NOT curr
                    if curr == u_e:
                        nxt = v_e
                    else:
                        nxt = u_e
                    vertices.append(nxt)
                    curr = nxt
                    
                paths.append(vertices)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
