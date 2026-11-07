import time

def construct():
    """
    Constructs a graph on 16 vertices (0..15) with diameter 2 where every edge is essential.
    The optimal such graph is the complete bipartite graph K_{8,8}, which has 64 edges.
    K_{8,8} is triangle-free. Removing any edge (u,v) with u in A, v in B results in
    distance 3 between u and v (no common neighbors), so diameter increases to 3.
    """
    start = time.time()
    deadline = start + 85
    
    # Partition vertices into two sets of 8
    A = range(8)
    B = range(8, 16)
    
    # Generate edges for K_{8,8}
    edges = [[u, v] for u in A for v in B]
    
    # Check deadline
    if time.time() > deadline:
        return edges
        
    return edges
