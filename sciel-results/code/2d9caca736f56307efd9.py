import time

def construct():
    """
    Constructs the optimal graph for the task: a complete bipartite graph K_{8,8}.
    
    Properties:
    1. Simple undirected graph on vertices 0..15.
    2. Diameter is exactly 2.
    3. Every edge is essential (removing any edge increases diameter to >= 3).
    4. Maximized edge count (64 edges), which is the theoretical maximum for a 
       triangle-free graph on 16 vertices (Turán's theorem).
    """
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s limit
    
    n = 16
    half = n // 2
    
    # Partition vertices into two sets A and B
    # A = {0, 1, ..., 7}
    # B = {8, 9, ..., 15}
    
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
            
    # Check deadline (though this static construction is instantaneous)
    if time.time() > deadline:
        # Fallback to a smaller valid graph if somehow too slow (unlikely)
        return [[0, i] for i in range(1, n)]
        
    return edges
