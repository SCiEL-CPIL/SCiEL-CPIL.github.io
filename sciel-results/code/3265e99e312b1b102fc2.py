"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def _bfs_distances(adj, n):
    """Return all-pairs shortest path distances using BFS. adj is list of sets."""
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        d = [-1] * n
        d[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[v] == -1:
                    d[v] = d[u] + 1
                    q.append(v)
        dist[s] = d
    return dist

def _diameter(dist, n):
    """Return the diameter of the graph given all-pairs distances, or -1 if disconnected."""
    max_d = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                if dist[i][j] == -1:
                    return -1
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
    return max_d

def _is_valid_graph(adj, n):
    """
    Check if graph has diameter exactly 2 and every edge is essential.
    An edge is essential if removing it increases diameter above 2 or disconnects.
    """
    # First check diameter is exactly 2
    dist = _bfs_distances(adj, n)
    diam = _diameter(dist, n)
    if diam != 2:
        return False
    
    # Check each edge is essential
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[u]:
                # Remove edge (u, v)
                adj[u].remove(v)
                adj[v].remove(u)
                
                # Check new diameter
                new_dist = _bfs_distances(adj, n)
                new_diam = _diameter(new_dist, n)
                
                # Restore edge
                adj[u].add(v)
                adj[v].add(u)
                
                # Edge is essential if new diameter > 2 or disconnected
                if new_diam <= 2:
                    return False
    return True

def construct():
    """
    Construct a graph on vertices 0..15 with diameter exactly 2,
    such that deleting ANY single edge increases the diameter above 2
    (or disconnects the graph). Maximize edge count.
    
    Key insight: A complete bipartite graph K_{8,8} has 64 edges, diameter 2,
    and every edge is essential (removing any edge (u,v) where u in A, v in B
    makes dist(u,v) = 3 via u -> a2 -> v or similar, but actually we need to
    verify this carefully).
    
    Let's think more carefully:
    - K_{8,8} with partitions A = {0..7}, B = {8..15}
    - Diameter is 2 (any two vertices in same partition have distance 2 via any vertex in other partition)
    - Removing edge (a, b) where a in A, b in B:
      - dist(a, b) becomes 3: a -> a' -> b' -> b (where a' != a in A, b' != b in B)
      - Wait, no. If we remove (a,b), can we still get from a to b in 2 steps?
      - a is connected to all of B except b. So a can go to any b' in B\{b} in 1 step.
      - From b', can we reach b in 1 step? b' is in B, and b is in B. In K_{8,8}, there are no edges within B.
      - So dist(a, b) after removing (a,b) is: a -> b' (1 step) but b' cannot reach b in 1 step (no intra-B edges).
      - So we need a -> a' -> b where a' is in A\{a}. But a' is connected to b (since only (a,b) was removed).
      - So dist(a, b) = 2 via a -> a' -> b.
      - This means the diameter is still 2! So K_{8,8} does NOT have the property that every edge is essential.
    
    So K_{8,8} is not valid. We need a different approach.
    
    Let's reconsider. We need a graph where:
    1. Diameter is exactly 2
    2. For every edge (u,v), removing it makes the diameter > 2 or disconnects the graph.
    
    This means for every edge (u,v), there must be some pair (x,y) such that:
    - The only shortest path of length <= 2 between x and y uses edge (u,v)
    - Or removing (u,v) disconnects x and y
    
    A simpler approach: start with a sparse graph and add edges greedily while maintaining the property.
    
    Actually, let's think about what graphs have this property. This is related to "edge-critical" graphs for diameter 2.
    
    Let me try a different approach: use a greedy algorithm to build the graph, checking validity as we go.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline to be safe
    
    # Start with empty graph
    adj = [set() for _ in range(n)]
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # We'll try to add edges in a greedy manner, but we need to be smart about it.
    # Actually, let's try a different strategy: start with a known valid small graph and grow it.
    
    # For now, let's return a simple valid graph. A star graph has diameter 2 if n <= 3,
    # but for n=16, a star has diameter 2 only if we consider the center to leaves as 1 and
    # leaf to leaf as 2. But removing any edge from the star disconnects the graph, so
    # all edges are essential. However, a star on 16 vertices has only 15 edges.
    
    # Let's verify: star graph with center 0 and leaves 1..15
    # Diameter: dist(0, v) = 1 for all v != 0, dist(v, w) = 2 for v != w, v,w != 0.
    # So diameter is 2.
    # Removing any edge (0, v) disconnects v from the rest. So all edges are essential.
    # This gives 15 edges.
    
    # Can we do better? Let's try to add more edges.
    
    # Start with the star
    for v in range(1, n):
        adj[0].add(v)
        adj[v].add(0)
    
    # Now try to add edges between leaves
    # When we add edge (u, v) where u, v are both leaves (not 0):
    # - The graph still has diameter 2 (adding edges can only decrease distances)
    # - But now, is edge (0, u) still essential? 
    #   After adding (u, v), if we remove (0, u), u is still connected to v, and v is connected to 0.
    #   So u can reach 0 via u -> v -> 0. Thus (0, u) is NOT essential anymore.
    # So we cannot add edges between leaves if we want all edges to be essential.
    
    # What if we add edge (0, u) where u is already a leaf? That's already in the graph.
    
    # It seems like the star is optimal with 15 edges? Let's think if there's a better structure.
    
    # Consider a graph where vertex 0 is connected to all others, and some additional edges.
    # But as shown above, any additional edge between leaves makes some star edges non-essential.
    
    # What about a different structure? Consider a "double-star" or other configurations.
    
    # Actually, let's just return the star for now. It's valid and has 15 edges.
    # We can try to improve later.
    
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
