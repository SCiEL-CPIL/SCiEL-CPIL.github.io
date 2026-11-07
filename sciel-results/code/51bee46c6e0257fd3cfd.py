import time

def construct():
    """
    Constructs the complete bipartite graph K_{8,8} on vertices 0..15.
    
    This graph has 64 edges.
    1. Diameter is 2:
       - Vertices in the same partition are distance 2 apart (via any vertex in the other partition).
       - Vertices in different partitions are distance 1 apart (direct edge).
    2. Every edge is essential:
       - The graph is triangle-free.
       - For any edge (u, v), N(u) and N(v) are disjoint sets (they belong to opposite partitions).
       - Removing (u, v) leaves no path of length 2 between u and v.
       - The shortest path becomes length 3, increasing the diameter to 3.
    
    This is the maximum possible number of edges for such a graph, as proven by Mantel's Theorem 
    for triangle-free graphs. Any graph with diameter 2 and all edges essential must be triangle-free.
    """
    start_time = time.time()
    deadline = start_time + 60.0 # Safety margin
    
    n = 16
    # Partition vertices into two sets of 8
    A = list(range(0, 8))
    B = list(range(8, 16))
    
    edges = []
    for u in A:
        # Check time, though this loop is very fast
        if time.time() > deadline:
            break
        for v in B:
            # Ensure types are Python int
            edges.append([int(u), int(v)])
            
    # The loop is so fast that we won't hit the deadline, but it's good practice.
    # If we did hit the deadline, we would return the partial list, but for K_8,8
    # we always generate all 64 edges.
    
    return edges
