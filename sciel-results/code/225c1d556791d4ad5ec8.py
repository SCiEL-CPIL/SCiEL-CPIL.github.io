import time
from collections import deque


def _check_girth(n, edges, g):
    """Return True iff no cycle is shorter than g (BFS from every vertex)."""
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] != -1:
                    if dist[u] + dist[v] + 1 < g:
                        return False
                else:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
    return True
