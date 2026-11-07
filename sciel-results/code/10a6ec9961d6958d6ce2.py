import time

def construct():
    """
    Constructs a graph on 16 vertices (0..15) with diameter 2 where every edge is essential.
    
    Reasoning:
    1. The condition "Deleting EACH individual edge must increase its diameter above two"
       implies that for every edge (u, v), the distance between u and v in the graph without
       that edge must be at least 3. This requires that u and v do not share any common
       neighbors. This property holds for all edges in a triangle-free graph.
    2. To maximize the number of edges in a triangle-free graph on n vertices, we use
       Mantel's Theorem, which states the maximum number of edges is floor(n^2 / 4),
       achieved by the complete bipartite graph K_{n/2, n/2} (for even n).
    3. For n=16, the optimal graph is K_{8,8}, which has 8 * 8 = 64 edges.
    4. K_{8,8} has diameter 2. Removing any edge increases the distance between its
       endpoints to 3, thus increasing the diameter to 3.
    
    The graph is constructed as the complete bipartite graph between vertices 0-7 and 8-15.
    """
    start = time.time()
    deadline = start + 5.0  # Safety margin, though this operation is instantaneous
    
    n = 16
    half = n // 2
    
    # Partition vertices into two sets of 8
    A = list(range(half))
    B = list(range(half, n))
    
    # Generate edges for K_{8,8}
    # Ensure types are Python int and list
    edges = []
    for u in A:
        for v in B:
            edges.append([int(u), int(v)])
            
    # Check deadline (just in case, though not strictly necessary for 64 edges)
    if time.time() > deadline:
        # Fallback: return what we have (which is complete)
        return edges
        
    return edges
