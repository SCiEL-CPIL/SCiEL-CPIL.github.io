"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin within 60s limit
    
    # The theoretical minimum number of distinct cycle lengths for a graph
    # with minimum degree at least 3 is 2.
    # A graph consisting of two disjoint K_{3,3} components achieves this.
    # - Min degree 3.
    # - Girth 4 (bipartite).
    # - Cycle lengths present: 4 and 6.
    # - Distinct cycle lengths: {4, 6}. Count = 2.
    
    edges = []
    
    # Check time before starting heavy work (though this is light)
    if time.time() > deadline:
        return edges

    # Component 1: K_{3,3} on vertices 0,1,2 (Left) and 3,4,5 (Right)
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # Check time
    if time.time() > deadline:
        return edges

    # Component 2: K_{3,3} on vertices 6,7,8 (Left) and 9,10,11 (Right)
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
