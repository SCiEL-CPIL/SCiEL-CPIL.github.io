"""Rewritten program: structural change to a compact two-hub solution (8 edges) with fast validation."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill

    n = P['n']

    # Structural change: use two hubs (6 and 17) instead of a single hub at 12.
    # This yields 8 added edges (vs 10 in the previous star topology).
    added_edges = [
        [0, 6], [2, 6], [4, 6],
        [13, 17], [15, 17], [19, 17], [21, 17], [23, 17]
    ]

    # Build adjacency matrix and edge set
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))

    # Add new edges
    for u, v in added_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            # Duplicate edge - invalid, return empty
            return []
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add(key)

    # Verify triangle-free
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                for w in range(v + 1, n):
                    if adj[u][w] and adj[v][w]:
                        return []

    # Verify diameter via BFS from each vertex
    def bfs_diameter(adj):
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d

    if time.time() > deadline:
        return []

    diam = bfs_diameter(adj)
    if diam <= 4:
        # Ensure all elements are Python int and list of lists
        result = [[int(u), int(v)] for u, v in added_edges]
        return result
    else:
        return []


# EVOLVE-BLOCK-END
