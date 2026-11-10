import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    n = 16
    # Partition: 0-7 and 8-15
    # Construct K(8,8)
    edges = []
    for a in range(8):
        for b in range(8, 16):
            edges.append([a, b])

    # Verify the graph
    if _verify(edges, n):
        return edges

    # Fallback: Star graph (valid but lower edge count)
    return [[0, v] for v in range(1, n)]

def _verify(edges, n):
    # Build adjacency list
    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Check diameter is exactly 2
    if _get_diameter(adj, n) != 2:
        return False

    # Check edge criticality
    for u, v in edges:
        # Remove edge
        adj[u].remove(v)
        adj[v].remove(u)

        d2 = _get_diameter(adj, n)

        # Restore edge
        adj[u].add(v)
        adj[v].add(u)

        # If diameter is still <= 2, edge is not critical
        if d2 is not None and d2 <= 2:
            return False

    return True

def _get_diameter(adj, n):
    max_d = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]
        i = 0
        while i < len(q):
            u = q[i]
            i += 1
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    q.append(w)
        if -1 in dist:
            return None  # Disconnected
        max_d = max(max_d, max(dist))
    return max_d
