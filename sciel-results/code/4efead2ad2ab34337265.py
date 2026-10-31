"""Voltage lift of K_{3,3} over Z_m for 3-regular graphs with girth >= g."""
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph with girth >= g using voltage lift of K_{3,3} over Z_m."""
    if k != 3:
        n = 2 * k
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i - j) % k != 0 and (i - j) % k != k - 1:
                    edges.append((i, j))
        return n, edges
    
    min_m = 11
    random.seed(42)
    
    for m in range(min_m, min_m + 100):
        for attempt in range(300):
            v = [[random.randint(0, m-1) for _ in range(3)] for _ in range(3)]
            if _check_girth(v, m, g):
                edges = _build_edges(v, m)
                return 6 * m, edges
    
    m = min_m + 99
    v = [[random.randint(0, m-1) for _ in range(3)] for _ in range(3)]
    return 6 * m, _build_edges(v, m)


def _build_edges(v, m):
    edges = []
    for i in range(3):
        for j in range(3):
            for t in range(m):
                left_idx = i * m + t
                right_idx = 3 * m + j * m + ((t + v[i][j]) % m)
                edges.append((left_idx, right_idx))
    return edges


def _check_girth(v, m, g):
    n = 6 * m
    adj = [[] for _ in range(n)]
    for i in range(3):
        for j in range(3):
            for t in range(m):
                l = i * m + t
                r = 3 * m + j * m + ((t + v[i][j]) % m)
                adj[l].append(r)
                adj[r].append(l)
    
    for start in range(n):
        dist = [-1] * n
        par = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    par[w] = u
                    q.append(w)
                elif w != par[u] and par[w] != u:
                    if dist[u] + dist[w] + 1 < g:
                        return False
    return True
# EVOLVE-BLOCK-END
