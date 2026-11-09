import time
from collections import deque

def _bfs_diameter(adj):
    """Return the diameter of the graph given by adjacency lists.
    Returns float('inf') if the graph is disconnected."""
    n = len(adj)
    diam = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for d in dist:
            if d == -1:
                return float('inf')
            if d > diam:
                diam = d
    return diam

def construct():
    start_time = time.time()
    n = 16
    half = n // 2
    edges = []
    
    # Construct K_{8,8} - the complete bipartite graph
    # This is triangle-free, so every edge is critical for diameter 2.
    # By Mantel's theorem, this is the maximum number of edges in a
    # triangle-free graph on 16 vertices (64 edges).
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    
    # Verify the result meets all requirements
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check diameter is exactly 2
    diam = _bfs_diameter(adj)
    if diam != 2:
        # Should not happen for K_{8,8}, but return empty if invalid
        return []
    
    # Verify every edge is critical: deleting it must increase diameter above 2 or disconnect
    # For K_{8,8}, removing any edge (u,v) means u and v have no common neighbor? 
    # Wait - in K_{8,8}, u is in {0..7} and v is in {8..15}.
    # Common neighbors of u and v: none, since u's neighbors are all in {8..15} and v's neighbors are all in {0..7}.
    # Actually wait: u's neighbors are all vertices in {8..15}. v's neighbors are all vertices in {0..7}.
    # So common neighbors of u and v: intersection of {8..15} and {0..7} = empty.
    # So after removing edge (u,v), dist(u,v) becomes infinite (disconnected? No!)
    # Actually u can still reach v via u -> w -> ... -> v? No.
    # u is in {0..7}, v is in {8..15}. u's neighbors are all in {8..15}. v's neighbors are all in {0..7}.
    # After removing (u,v): u can reach any w in {8..15} (w != v) directly. From w, w can reach any x in {0..7} directly.
    # So u -> w -> x where x is in {0..7}. But can u reach v? 
    # u -> w (w in {8..15}, w != v) -> x (x in {0..7}). Then x -> v? x's neighbors are all in {8..15}. So x -> v works!
    # So dist(u,v) = 3 after removing (u,v). Diameter becomes 3 > 2. Good.
    
    # Let's verify with a quick check on a few edges to be safe
    # But given the theoretical guarantee, we can trust it.
    
    # Safety deadline check
    if time.time() - start_time > 55:
        return edges
        
    # Return edges as list of [u, v] with Python ints
    result = []
    for u, v in edges:
        result.append([int(u), int(v)])
    return result
