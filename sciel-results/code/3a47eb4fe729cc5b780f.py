# EVOLVE-BLOCK-START

import time

def construct():
    """
    Constructs a graph on 16 vertices (0..15) with diameter 2 where every edge is essential.
    The optimal such graph is the complete bipartite graph K_{8,8}, which is triangle-free.
    In a triangle-free graph with diameter 2, every edge is essential because removing any edge
    (u,v) leaves no common neighbor, so d(u,v) increases from 1 to 3.
    Max edges = floor(16^2 / 4) = 64.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 16
    # Partition vertices into two sets of size 8
    set_a = list(range(8))
    set_b = list(range(8, 16))
    
    edges = []
    for u in set_a:
        if time.time() > deadline:
            break
        for v in set_b:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
