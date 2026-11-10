"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

N = 16
DEADLINE = 85.0  # safety margin before 90s kill

def bfs_diameter(adj, n):
    """Return the diameter of the graph given adjacency list.
    If graph is disconnected, return -1."""
    diam = 0
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
            return -1
        d = max(dist)
        if d > diam:
            diam = d
    return diam

def check_validity(edges, n):
    """Check if edges form a valid graph with diameter exactly 2 and every edge is essential.
    Returns True if valid, False otherwise."""
    m = len(edges)
    # Build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
        adj[a].append(b)
        adj[b].append(a)
    
    # Check diameter is exactly 2
    if bfs_diameter(adj, n) != 2:
        return False
    
    # Check each edge is essential (removing it disconnects or increases diameter above 2)
    for i, (u, v) in enumerate(edges):
        adj2 = [lst[:] for lst in adj]
        adj2[u].remove(v)
        adj2[v].remove(u)
        if bfs_diameter(adj2, n) <= 2:
            return False
    return True

def construct():
    """Construct a dense graph on 16 vertices with diameter exactly 2 where every edge is essential."""
    start_time = time.time()
    
    # Strategy: Start with a complete graph and remove edges while maintaining the property.
    # Actually, we need to find a graph where EVERY edge is essential.
    # This means the graph is a "2-edge-connected diameter-2 graph" in a strong sense.
    
    # Let's think about this differently. We want to maximize edge count.
    # A complete graph K_16 has diameter 1, not 2. So we need to remove edges to make diameter 2.
    # But we need every remaining edge to be essential.
    
    # Alternative approach: Start with a sparse graph that has diameter 2, then add edges
    # as long as the property is maintained. But adding edges might make some other edge non-essential.
    
    # Let's try a greedy approach:
    # 1. Start with a base graph that has diameter 2.
    # 2. Try adding edges one by one, checking if the property still holds.
    
    # First, let's construct a base graph. A star graph has diameter 2, but removing any edge
    # disconnects it. So a star on 16 vertices has 15 edges and every edge is essential.
    # But we want to maximize edge count.
    
    # Let's try to build up from a star. Start with edges from 0 to all others.
    # Then try to add edges between non-adjacent vertices (excluding 0).
    
    def try_add_edge(edges, adj, n):
        """Try to add an edge between two non-adjacent vertices if it maintains the property."""
        # This is expensive, so we'll use it carefully.
        pass
    
    # Let's use a different approach: start with a known valid graph and try to improve.
    # Base: star graph (15 edges)
    edges = [[0, v] for v in range(1, N)]
    
    # Try to add edges between vertices 1-15
    # For each pair (i, j) where i < j, i, j >= 1, try adding edge (i, j)
    # Check if the resulting graph still has the property.
    
    # To speed up, we'll maintain the adjacency list and use a faster check.
    
    def is_valid_fast(adj, n, edge_list):
        """Fast validity check using the precomputed adjacency."""
        # Check diameter is 2
        if bfs_diameter(adj, n) != 2:
            return False
        # Check each edge is essential
        for u, v in edge_list:
            adj2 = [lst[:] for lst in adj]
            adj2[u].remove(v)
            adj2[v].remove(u)
            if bfs_diameter(adj2, n) <= 2:
                return False
        return True
    
    # Try adding edges
    for i in range(1, N):
        for j in range(i+1, N):
            if time.time() - start_time > DEADLINE:
                break
            # Check if (i, j) is already an edge
            if j in adj[i]:
                continue
            # Try adding edge (i, j)
            new_edges = edges + [[i, j]]
            new_adj = [lst[:] for lst in adj]
            new_adj[i].append(j)
            new_adj[j].append(i)
            if is_valid_fast(new_adj, N, new_edges):
                edges = new_edges
                adj = new_adj
        if time.time() - start_time > DEADLINE:
            break
    
    # Return the best graph found
    # Convert to list of [u, v] with Python ints
    result = []
    for u, v in edges:
        result.append([int(u), int(v)])
    return result
# EVOLVE-BLOCK-END
