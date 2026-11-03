import random
import time
import math

def _construct_luw(k, g):
    # ... (copy of seed code) ...
    pass

def _check_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    for start in range(n):
        visited = {start: 0}
        queue = [(start, -1, 0)]
        while queue:
            u, parent, dist = queue.pop(0)
            for v in adj[u]:
                if v == parent:
                    continue
                if v in visited:
                    # Cycle found
                    # Length is dist + 1 + dist_to_v?
                    # We need to store the path or distance from start
                    # This simple BFS doesn't track the other end's distance easily.
                    # Better: BFS from start, if we reach a visited node, 
                    # we need to know the distance of that node from start.
                    # If visited[v] is dist_v, and current dist is dist_u,
                    # cycle length is dist_u + dist_v + 1.
                    # But this might not be the shortest cycle through u,v.
                    # For girth check, we just need to know if ANY cycle < g exists.
                    # A BFS tree from start finds the shortest path from start to all nodes.
                    # If we find a back edge, the cycle length is dist[u] + dist[v] + 1.
                    # This is the length of the cycle formed by the tree paths + back edge.
                    # This is a valid cycle. If its length < g, fail.
                    if visited[v] + dist + 1 < g:
                        return False
                else:
                    visited[v] = dist + 1
                    queue.append((v, u, dist + 1))
    return True

def _try_random_graph(k, n, g, time_budget):
    start = time.time()
    # Try a few random restarts
    for _ in range(10):
        if time.time() - start > time_budget:
            return None
        # Generate random regular graph
        edges = _generate_random_regular(k, n)
        if edges is None:
            continue
        if _check_girth(n, edges, g):
            return edges
    return None
