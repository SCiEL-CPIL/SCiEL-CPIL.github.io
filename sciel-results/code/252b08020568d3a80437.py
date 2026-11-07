import time
import itertools

P = {'n': 16}

def _verify(edges, n):
    """
    Verify that:
    1. The graph has diameter exactly 2.
    2. Deleting EACH individual edge increases the diameter above 2 or disconnects the graph.
    """
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    # Check for self-loops and duplicates
    if len(edge_set) != len(edges):
        return False
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
    
    # BFS function to compute all-pairs distances
    def all_pairs_bfs(adj):
        dist = [[-1] * n for _ in range(n)]
        for s in range(n):
            dist[s][s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[s][v] == -1:
                        dist[s][v] = dist[s][u] + 1
                        queue.append(v)
        return dist
    
    # Check diameter is exactly 2
    dist = all_pairs_bfs(adj)
    max_dist = 0
    for i in range(n):
        for j in range(i+1, n):
            if dist[i][j] == -1:
                return False  # Disconnected
            if dist[i][j] > max_dist:
                max_dist = dist[i][j]
    if max_dist != 2:
        return False
    
    # Check that deleting each edge increases diameter or disconnects
    for (u, v) in edge_set:
        # Remove edge (u, v)
        adj[u].discard(v)
        adj[v].discard(u)
        
        dist2 = all_pairs_bfs(adj)
        
        # Restore edge
        adj[u].add(v)
        adj[v].add(u)
        
        # Check if diameter > 2 or disconnected
        max_dist2 = 0
        disconnected = False
        for i in range(n):
            for j in range(i+1, n):
                if dist2[i][j] == -1:
                    disconnected = True
                    break
                if dist2[i][j] > max_dist2:
                    max_dist2 = dist2[i][j]
            if disconnected:
                break
        
        if not disconnected and max_dist2 <= 2:
            return False  # Edge is not critical
    
    return True

def construct():
    """
    Returns a list of edges for the complete bipartite graph K_{8,8}.
    
    Reasoning:
    1. The condition that deleting any edge increases the diameter above 2 (or disconnects)
       implies that no edge can be part of a triangle, unless its removal disconnects the graph.
       In a dense graph maximizing edges, we assume high connectivity, so edges are not bridges.
       Thus, the graph must be triangle-free.
    2. By Mantel's Theorem, the maximum number of edges in a triangle-free graph on n vertices
       is floor(n^2 / 4).
    3. For n=16, max edges = 64, achieved by K_{8,8}.
    4. K_{8,8} has diameter 2.
    5. In K_{8,8}, any edge (u,v) connects partitions V1 and V2. N(u) subset V2, N(v) subset V1.
       N(u) intersect N(v) is empty.
       Removing (u,v) removes the direct link. No path of length 2 exists (requires common neighbor).
       Shortest path becomes length 3 (u -> w1 -> w2 -> v).
       Diameter increases from 2 to 3.
       Condition satisfied.
    """
    n = P['n']
    
    if n < 2:
        return []
    
    # Split vertices into two partitions as evenly as possible
    half = n // 2
    part1 = list(range(half))
    part2 = list(range(half, n))
    
    edges = []
    
    # Construct all cross edges
    for u in part1:
        for v in part2:
            edges.append([u, v])
    
    return edges
