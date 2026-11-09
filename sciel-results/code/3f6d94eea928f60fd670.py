import time

def construct():
    """
    Construct a simple undirected graph on vertices 0..14 with 101 edges
    that avoids equal-degree path endpoints for all simple 3-edge paths.
    
    Construction:
    - Partition: A = {0..6}, B = {7..14}
    - All cross edges between A and B (complete bipartite K_{7,8}): 56 edges
    - All edges within A (complete graph K_7): 21 edges
    - Edges within B: K_8 minus a perfect matching of 4 edges: 24 edges
    - Total: 56 + 21 + 24 = 101 edges
    
    Degrees:
    - Vertices in A: 8 (cross) + 6 (within A) = 14
    - Vertices in B: 7 (cross) + 6 (within B, since K_8 minus matching gives degree 6) = 13
    
    Since all vertices in A have degree 14 and all vertices in B have degree 13,
    the degree sets are disjoint, so no two vertices with equal degree can be
    endpoints of a 3-edge path.
    """
    n = 15
    A = list(range(7))    # 0..6
    B = list(range(7, 15))  # 7..14
    
    edges = []
    
    # All cross edges between A and B
    for a in A:
        for b in B:
            edges.append([a, b])
    
    # All edges within A (K_7)
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            edges.append([A[i], A[j]])
    
    # Edges within B: K_8 minus a perfect matching
    # Perfect matching in B: (7,8), (9,10), (11,12), (13,14)
    matching = [(7, 8), (9, 10), (11, 12), (13, 14)]
    matching_set = set()
    for u, v in matching:
        matching_set.add((u, v))
        matching_set.add((v, u))
    
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            u, v = B[i], B[j]
            if (u, v) not in matching_set:
                edges.append([u, v])
    
    # Return as list of [u, v] with Python int types
    return [[int(u), int(v)] for u, v in edges]
