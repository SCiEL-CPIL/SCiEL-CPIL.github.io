import time

def _is_rainbow(adj, n):
    """Return True if every 4-cycle in the graph (adjacency with colour dict) is rainbow."""
    adj_set = [set() for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            adj_set[u].add(v)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # common neighbours of u and v
            common = adj_set[u] & adj_set[v]
            for w in common:
                for x in common:
                    if w >= x:
                        continue
                    # 4-cycle u-v-w-x-u
                    cols = [adj[u][v], adj[v][w], adj[w][x], adj[x][u]]
                    if len(set(cols)) != 4:
                        return False
    return True

def construct():
    deadline = time.time() + 120.0
    n = 20
    # Primary construction: complete graph, edge (u,v) coloured by min(u,v).
    # Each colour c is used only on edges incident to vertex c (a star),
    # so no two same-coloured edges are disjoint -> every 4-cycle is rainbow.
    adj = [dict() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = u  # colour by the smaller-index endpoint
            adj[u][v] = c
            adj[v][u] = c
    edges = []
    for u in range(n):
        for v, c in adj[u].items():
            if v > u:
                edges.append([u, v, c])

    # Verify the primary construction; if somehow invalid, fall back to greedy.
    if not _is_rainbow(adj, n):
        # Greedy fallback: build a graph with star-colour classes.
        adj2 = [dict() for _ in range(n)]
        # colour assignment: vertex i is centre of colour i
        order = list(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                c = i
                adj2[i][j] = c
                adj2[j][i] = c
        edges = []
        for u in range(n):
            for v, c in adj2[u].items():
                if v > u:
                    edges.append([u, v, c])
        if _is_rainbow(adj2, n):
            return edges
        # Last resort: star
        return [[0, v, 0] for v in range(1, n)]
    return edges
