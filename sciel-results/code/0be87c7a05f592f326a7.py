"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
        
        # Build adjacency lists for efficient cost matrix construction
        # For each vertex, list of edge indices starting at that vertex
        edges_starting = {}
        for i, (u, v) in enumerate(edges):
            edges_starting.setdefault(u, []).append(i)
        
        m = len(edges)
        
        # Cost matrix: left i (edge i ends at v_i) matched to right j (edge j starts at u_j)
        # Cost 0 if edges[i][1] == edges[j][0], else BIG
        BIG = 10**9
        cost = np.full((m, m), BIG, dtype=np.int64)
        
        for i in range(m):
            end_i = edges[i][1]
            for j in edges_starting.get(end_i, []):
                cost[i][j] = 0
        
        # Find minimum cost perfect matching
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Build the paths
        next_edge = [-1] * m
        prev_edge = [-1] * m
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] == 0:
                next_edge[r] = c
                prev_edge[c] = r
        
        # Find all path starts (edges with no predecessor)
        paths = []
        for i in range(m):
            if prev_edge[i] == -1:
                # Start a new path at edge i
                path_edges = [i]
                cur = i
                while next_edge[cur] != -1:
                    cur = next_edge[cur]
                    path_edges.append(cur)
                # Convert edge indices to vertex list
                vertices = [edges[path_edges[0]][0]]
                for e_idx in path_edges:
                    vertices.append(edges[e_idx][1])
                paths.append(vertices)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
