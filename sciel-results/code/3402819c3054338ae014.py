import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 60.0  # Seconds

    # The known optimal structure for minimizing distinct cycle lengths
    # with min degree 3 and triangle-free (girth >= 4) on 12 vertices
    # is two disjoint copies of K_{3,3}.
    # K_{3,3} has cycle lengths 4 and 6.
    # Two disjoint copies still only have cycle lengths 4 and 6.
    # Total distinct lengths = 2.
    
    # Define edges for the first K_{3,3} on vertices 0..5
    # Partition A: {0, 1, 2}, Partition B: {3, 4, 5}
    edges = set()
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.add((u, v))
            
    # Define edges for the second K_{3,3} on vertices 6..11
    # Partition A: {6, 7, 8}, Partition B: {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.add((u, v))

    # Convert to sorted list of lists for consistent output
    # Ensure types are standard Python int and list
    result_edges = []
    for u, v in edges:
        if u > v:
            u, v = v, u
        result_edges.append([int(u), int(v)])
    
    # Sort for deterministic output
    result_edges.sort()

    # Local Validation to ensure no broken rules
    # 1. Check types and basic structure
    if not isinstance(result_edges, list):
        return []
        
    n = 12
    deg = [0] * n
    adj = {i: set() for i in range(n)}
    
    for e in result_edges:
        if len(e) != 2:
            return []
        u, v = e
        if not isinstance(u, int) or not isinstance(v, int):
            return []
        if u == v:
            return []
        if u >= n or v >= n or u < 0 or v < 0:
            return []
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)

    # 2. Check minimum degree >= 3
    for i in range(n):
        if deg[i] < 3:
            return []

    # 3. Check triangle-free (girth >= 4)
    # In a simple graph, triangles are the only cycles of length 3.
    # We check for common neighbors between any edge endpoints.
    for u in range(n):
        for v in adj[u]:
            if u < v:
                # Check if any common neighbor w forms a triangle u-v-w
                # w must be neighbor of both u and v
                # If w != u and w != v, and w is in adj[u] and adj[v], triangle exists.
                # Since graph is simple, u is not in adj[u].
                common = adj[u].intersection(adj[v])
                if common:
                    return []

    # 4. Check that the graph contains at least one cycle
    # A graph with n vertices and m edges has a cycle if m >= n.
    # Here m = 18, n = 12. 18 >= 12, so cycles exist.
    # Specifically, K_{3,3} is connected and has cycles.
    
    # 5. Check disconnected components allowed? Yes.
    
    # The raw value is the number of distinct simple cycle lengths.
    # For two K_{3,3}s, lengths are 4 and 6. Count = 2.
    # This is the minimal possible.
    
    if time.time() - start_time > deadline:
        return [] # Should not happen, but guard just in case

    return result_edges
