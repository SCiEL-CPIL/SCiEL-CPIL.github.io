"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with 12 vertices, minimum degree 3, and girth at least 4.
    Uses two disjoint copies of the complete bipartite graph K_{3,3}.
    K_{3,3} has cycles of length 4 and 6 only.
    Thus, the number of distinct cycle lengths is 2.
    
    The design is kept within the K_{3,3} family. Verification ensures
    the graph is simple, undirected, triangle-free, and every vertex has
    degree at least 3. The structure is deterministic and fast.
    """
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Partition A: {0, 1, 2}, Partition B: {3, 4, 5}
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    # Partition A: {6, 7, 8}, Partition B: {9, 10, 11}
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    # Verify constraints (defensive check)
    from collections import defaultdict
    degree = defaultdict(int)
    edge_set = set()
    for u, v in edges:
        if u == v:
            raise ValueError("Loop detected")
        if (v, u) in edge_set:
            raise ValueError("Duplicate edge detected")
        edge_set.add((u, v))
        edge_set.add((v, u))
        degree[u] += 1
        degree[v] += 1
    
    for i in range(12):
        if degree[i] < 3:
            raise ValueError(f"Vertex {i} has degree {degree[i]} < 3")
    
    # Check triangle-free (girth >= 4)
    # For K_{3,3}, there are no triangles.
    # We can verify by checking if any edge (u,v) has a common neighbor w.
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    for u in range(12):
        for v in adj[u]:
            if u > v:  # Check each edge once
                # If they share a neighbor, it's a triangle
                if adj[u].intersection(adj[v]):
                    raise ValueError("Triangle detected")
    
    return edges

# EVOLVE-BLOCK-END
