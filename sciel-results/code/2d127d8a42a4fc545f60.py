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
        # Each edge is an "item". We want to chain them: edge (a,b) can be followed by edge (b,c)
        # Build bipartite graph: left = edges (as tails), right = edges (as heads)
        # Left node i represents edge e_i = (u_i, v_i) as a "tail" (ends at v_i)
        # Right node j represents edge e_j = (u_j, v_j) as a "head" (starts at u_j)
        # Add edge from left i to right j if v_i == u_j (can chain e_i then e_j)
        # Cost: 0 if chainable, large if not. We want max matching with min cost.
        m = len(edges)
        # Build adjacency: for each edge index i, which edge indices j can follow it
        # edges[i] = (u, v), so it ends at v. It can be followed by edge j if edges[j][0] == v
        # But also edges[i] starts at u, so it can be preceded by edge j if edges[j][1] == u
        # For chaining: ... -> e_j -> e_i -> ... means e_j ends at start of e_i
        # e_j = (a, b), e_i = (b, c): chain if e_j[1] == e_i[0]
        
        # Left side: each edge as a "start" node (we'll match a predecessor to a successor)
        # Actually: left node i = edge i as a "tail end" (ends at edges[i][1])
        # right node j = edge j as a "head start" (starts at edges[j][0])
        # Match left i to right j if edges[i][1] == edges[j][0], meaning edge i can be immediately followed by edge j
        
        # Cost matrix: 0 if chainable, BIG if not
        BIG = 10**9
        cost = np.full((m, m), BIG, dtype=np.int64)
        for i in range(m):
            end_i = edges[i][1]
            for j in range(m):
                if edges[j][0] == end_i:
                    cost[i][j] = 0
        
        # Find minimum cost perfect matching
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Number of valid matches (cost 0)
        valid_matches = sum(1 for r, c in zip(row_ind, col_ind) if cost[r, c] == 0)
        
        # Build the paths
        # next_edge[i] = j if edge i is followed by edge j, else -1
        # prev_edge[j] = i if edge i is followed by edge j, else -1
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
