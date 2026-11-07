import time

def construct():
    """
    Constructs a simple undirected graph on vertices 0..15 with diameter exactly 2,
    where every edge is essential.
    
    Strategy:
    Use the complete bipartite graph K_{8,8}.
    Partition A: 0..7
    Partition B: 8..15
    Edges: all (u, v) with u in A, v in B.
    
    Properties:
    1. Diameter is 2.
    2. Triangle-free.
    3. Every edge is essential because removing any edge (u, v) increases the distance
       between u and v to 3 (since no common neighbor exists in a bipartite graph 
       for adjacent vertices, wait, u in A, v in B. Neighbors of u are in B. 
       Neighbors of v are in A. No common neighbor. So d(u, v) becomes 3).
    
    This maximizes the edge count among triangle-free graphs with diameter 2,
    which is the likely optimal family for this constraint.
    """
    start_time = time.time()
    deadline = 60.0  # Safety margin for 90s limit
    
    N = 16
    mid = N // 2
    
    edges = []
    # Generate edges for K_{8,8}
    for u in range(mid):
        for v in range(mid, N):
            edges.append([u, v])
            
    # Check time (should be instantaneous)
    if time.time() - start_time > deadline:
         # Fallback, though unlikely
         return edges
         
    return edges
