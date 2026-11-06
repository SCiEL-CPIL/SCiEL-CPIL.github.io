"""Path decomposition using minimum-cost maximum matching on edge-chain graph."""

def construct():
    import time
    import numpy as np
    from random import Random
    from scipy.optimize import linear_sum_assignment

    deadline = time.time() + 130

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        m = len(edges)
        if m == 0:
            return []
        if m == 1:
            return [list(edges[0])]

        # Build cost matrix: C[i][j] = 0 if edges[i] and edges[j] share a vertex, else BIG
        BIG = 10**9
        C = np.full((m, m), BIG, dtype=np.float64)
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(m):
                if i == j:
                    continue
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    C[i][j] = 0.0

        # Find minimum cost perfect matching (assignment)
        row_ind, col_ind = linear_sum_assignment(C)

        # Build next/prev from matching
        next_edge = {}
        prev_edge = {}
        for r, c in zip(row_ind, col_ind):
            if C[r, c] == 0:
                next_edge[r] = c
                prev_edge[c] = r

        # Find all maximal paths (chains)
        paths = []
        # A node is a "start" if it has no predecessor
        starts = [i for i in range(m) if i not in prev_edge]

        for start in starts:
            # Follow the chain
            edge_chain = [start]
            current = start
            while current in next_edge:
                current = next_edge[current]
                edge_chain.append(current)

            # Convert edge chain to vertex path
            # For each consecutive pair of edges, they share a vertex.
            # We need to determine the order of vertices.
            vertex_path = []
            for idx, e_idx in enumerate(edge_chain):
                u, v = edges[e_idx]
                if idx == 0:
                    vertex_path.append(u)
                    vertex_path.append(v)
                else:
                    prev_u, prev_v = edges[edge_chain[idx - 1]]
                    # The shared vertex between prev edge and current edge
                    # Determine which endpoint of current edge connects to prev edge
                    if prev_v == u:
                        vertex_path.append(v)  # u is shared, add v
                    elif prev_v == v:
                        vertex_path.append(u)  # v is shared, add u
                    elif prev_u == u:
                        vertex_path.append(v)
                    else:  # prev_u == v
                        vertex_path.append(u)
            paths.append(vertex_path)

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue

        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)

    return results
