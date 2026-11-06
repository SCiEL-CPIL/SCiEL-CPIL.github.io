"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = 110.0
    n = 40
    
    def time_ok():
        return time.time() - start_time < deadline
    
    def has_c6(graph, u, v):
        """Check if adding edge (u,v) creates a C6.
        Returns True if there's a simple path of length 5 from u to v in graph."""
        dist = {u: 0}
        queue = deque([u])
        while queue:
            node = queue.popleft()
            if dist[node] >= 5:
                continue
            for nb in graph[node]:
                if nb == v:
                    if dist[node] + 1 == 5:
                        return True
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        return dist.get(v, -1) == 5
    
    # Phase 1: Greedy construction
    random.seed(42)
    graph = {i: set() for i in range(n)}
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if not time_ok():
            break
        if not has_c6(graph, u, v):
            graph[u].add(v)
            graph[v].add(u)
    
    # Phase 2: Hill climbing - try to add edges
    for _ in range(5):
        if not time_ok():
            break
        improved = True
        while improved and time_ok():
            improved = False
            for u in range(n):
                if not time_ok():
                    break
                for v in range(u+1, n):
                    if not time_ok():
                        break
                    if v not in graph[u] and not has_c6(graph, u, v):
                        graph[u].add(v)
                        graph[v].add(u)
                        improved = True
    
    edges = []
    for u in range(n):
        for v in graph[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
