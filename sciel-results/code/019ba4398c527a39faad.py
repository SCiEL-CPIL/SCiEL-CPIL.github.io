import time

def construct():
    """
    Constructs a graph on 16 vertices (0..15) with diameter 2 where every edge is essential.
    The optimal such graph is the complete bipartite graph K_{8,8}, which has 64 edges.
    K_{8,8} is triangle-free, so removing any edge (u,v) eliminates the direct connection 
    and leaves no length-2 path between u and v (as N(u) and N(v) are disjoint in the 
    remaining graph for the specific pair u,v? No, N(u) and N(v) are disjoint sets 
    of vertices in the original graph because they are in different partitions. 
    Wait. In K_{8,8}, u in A, v in B. N(u) = B \ {v} is not correct. N(u) = B. 
    N(v) = A. 
    Intersection N(u) ∩ N(v) = B ∩ A = empty set. 
    So there is no common neighbor. 
    Thus no path of length 2. 
    Distance becomes 3. 
    Diameter becomes 3. 
    Condition holds.
    """
    start = time.time()
    deadline = start + 85
    
    n = 16
    # Partition vertices into two sets of 8
    A = list(range(0, 8))
    B = list(range(8, 16))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
            
    # The graph is K_{8,8}. 
    # It is simple, undirected.
    # Diameter is 2.
    # Every edge is critical because the graph is triangle-free.
    
    # Check deadline just in case, though generation is instant
    if time.time() > deadline:
        return edges
        
    return edges
