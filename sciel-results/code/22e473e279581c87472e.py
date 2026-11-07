"""
Program for the task: Dense graphs with every edge essential for diameter two.
Constructs a graph on vertices 0..15 (n=16) with maximum edges such that:
1. Diameter is exactly 2.
2. Deleting any single edge increases diameter > 2 or disconnects the graph.
"""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 16}

def _bfs_diameter(adj, n):
    """Compute diameter via BFS from each vertex. Returns (diam, connected)."""
    max_dist = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if -1 in dist:
            return (float('inf'), False)
        d = max(dist)
        if d > max_dist:
            max_dist = d
    return (max_dist, True)

def _is_edge_essential(adj, n, u, v):
    """Check if removing edge (u,v) makes diameter > 2 or disconnects."""
    # Remove edge
    adj[u].remove(v)
    adj[v].remove(u)
    diam, connected = _bfs_diameter(adj, n)
    # Restore
    adj[u].append(v)
    adj[v].append(u)
    if not connected:
        return True
    return diam > 2

def construct():
    """
    Construct the maximum-edge graph on 16 vertices with diameter exactly 2
    where every edge is essential (removing any edge increases diameter > 2 or disconnects).
    
    Strategy: Start with K_{8,8} (64 edges, triangle-free, diameter 2).
    Verify each edge is essential. If K_{8,8} passes, return it.
    Otherwise, greedily add edges from the complement while maintaining validity.
    """
    n = 16
    deadline = time.time() + 80  # safety margin before 90s kill

    # Build adjacency for K_{8,8}
    half = n // 2
    part1 = list(range(half))
    part2 = list(range(half, n))
    
    adj = [[] for _ in range(n)]
    edges = []
    for u in part1:
        for v in part2:
            adj[u].append(v)
            adj[v].append(u)
            edges.append([u, v])
    
    # Verify K_{8,8} is valid
    diam, connected = _bfs_diameter(adj, n)
    if not connected or diam != 2:
        # Fallback: shouldn't happen
        return []
    
    # Check if every edge is essential
    all_essential = True
    for e in edges:
        u, v = e[0], e[1]
        if not _is_edge_essential(adj, n, u, v):
            all_essential = False
            break
    
    if all_essential:
        # K_{8,8} is valid - return it
        return [[int(u), int(v)] for u, v in edges]
    
    # K_{8,8} is not valid. Need to find a valid subgraph or different graph.
    # Try: remove edges from K_{8,8} until all remaining edges are essential,
    # but this reduces edge count. Instead, try a different construction.
    
    # Alternative: Use a known edge-minimal diameter-2 graph and try to add edges.
    # Start with a cycle C_16 plus some chords to get diameter 2.
    # Actually, let's try: start empty, add edges greedily maintaining validity.
    
    # For now, return a valid but smaller graph as fallback.
    # A path-based construction: use two centers.
    # Center 0 connected to all others (15 edges). 
    # But that's a star, diameter 2, but edge (0,v): removing it, v is disconnected? No, v has no other edges.
    # So star IS valid! Each edge is a bridge.
    # Star K_{1,15}: 15 edges. Diameter 2. Every edge is a bridge, so removing disconnects.
    # That's valid but only 15 edges.
    
    # Better: try to find a denser valid graph.
    # Let's try: two hubs 0 and 1. 0 connected to 2..15, 1 connected to 2..15, 0-1 edge.
    # That's 14+14+1 = 29 edges. Check if valid.
    adj2 = [[] for _ in range(n)]
    edges2 = []
    for i in range(2, n):
        adj2[0].append(i)
        adj2[i].append(0)
        edges2.append([0, i])
        adj2[1].append(i)
        adj2[i].append(1)
        edges2.append([1, i])
    adj2[0].append(1)
    adj2[1].append(0)
    edges2.append([0, 1])
    
    diam2, conn2 = _bfs_diameter(adj2, n)
    if conn2 and diam2 == 2:
        valid2 = True
        for e in edges2:
            u, v = e[0], e[1]
            if not _is_edge_essential(adj2, n, u, v):
                valid2 = False
                break
        if valid2:
            return [[int(u), int(v)] for u, v in edges2]
    
    # Fallback: star
    adj3 = [[] for _ in range(n)]
    edges3 = []
    for i in range(1, n):
        adj3[0].append(i)
        adj3[i].append(0)
        edges3.append([0, i])
    return [[int(u), int(v)] for u, v in edges3]
# EVOLVE-BLOCK-END
