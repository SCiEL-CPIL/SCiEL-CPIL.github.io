import time
import random
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

def _is_edge_critical(adj, edge):
    """Check if removing 'edge' from the graph increases diameter or disconnects."""
    u, v = edge
    n = len(adj)
    
    # Create a copy of adj without the edge
    # For efficiency, we can simulate BFS on the fly or modify a copy
    # Given n=16, copy is cheap
    
    adj_copy = [list(neigh) for neigh in adj]
    # Remove u from v's list and v from u's list
    if v in adj_copy[u]:
        adj_copy[u].remove(v)
    if u in adj_copy[v]:
        adj_copy[v].remove(u)
        
    return _bfs_diameter(adj_copy) > 2

def construct():
    t0 = time.time()
    n = 16
    
    # Start with complete graph
    adj = [[j for j in range(n) if j != i] for i in range(n)]
    
    # To ensure we find a good solution, we might need to randomize the removal order
    # But we want to MAXIMIZE edges.
    # So we should remove edges that are "most redundant"?
    # Or just remove any redundant edge.
    
    # Let's try a greedy approach:
    # 1. Check if current graph is valid (all edges critical).
    # 2. If not, find a redundant edge and remove it.
    # 3. Repeat until valid or no more edges to remove (should not happen if we start with K16 and stop if diam>2).
    
    # However, removing an edge might make the diameter > 2.
    # If so, we cannot remove it.
    # But if it's redundant, removing it DOES NOT increase diameter.
    # So if it's redundant, removing it keeps diameter 2.
    # So we can safely remove it.
    
    # Wait, if we remove a redundant edge, the graph changes.
    # Other edges might become redundant or critical.
    
    while time.time() - t0 < 50:
        # Check if all edges are critical
        all_critical = True
        redundant_edge = None
        
        # Iterate through all edges
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    # Check if edge (u, v) is critical
                    if not _is_edge_critical(adj, (u, v)):
                        all_critical = False
                        redundant_edge = (u, v)
                        break
            if not all_critical:
                break
        
        if all_critical:
            # Found a valid graph
            break
        
        # Remove the redundant edge
        if redundant_edge:
            u, v = redundant_edge
            adj[u].remove(v)
            adj[v].remove(u)
        else:
            # No redundant edge found but not all critical? 
            # This shouldn't happen if logic is correct.
            break

    # Convert to edge list
    edge_list = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_list.append([int(u), int(v)])
    
    # Verify diameter is exactly 2
    if _bfs_diameter(adj) != 2:
        # Fallback to a known valid graph if this fails
        # e.g. Clebsch graph? But Clebsch is not critical.
        # Let's return the best we have.
        pass
        
    return edge_list

# To improve performance, we can pre-calculate common neighbors
# and use a faster criticality check.

def _common_neighbors_count(adj, u, v):
    """Count common neighbors of u and v."""
    set_u = set(adj[u])
    set_v = set(adj[v])
    return len(set_u.intersection(set_v))

def _is_edge_critical_fast(adj, edge):
    u, v = edge
    n = len(adj)
    
    # Remove edge
    adj_copy = [list(neigh) for neigh in adj]
    adj_copy[u].remove(v)
    adj_copy[v].remove(u)
    
    # Check diameter
    # We only need to check if any pair has distance > 2
    # In the original graph, all pairs had dist <= 2.
    # Only pairs that had dist 2 and relied on (u, v) might change.
    # But it's easier to just run BFS.
    
    return _bfs_diameter(adj_copy) > 2
