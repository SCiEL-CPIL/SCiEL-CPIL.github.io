import time
import random

def construct():
    """
    Constructs a K4-free graph on 18 vertices that minimizes the size of the 
    largest triangle-free induced subgraph.
    
    The optimal structure is believed to be the complete tripartite graph K_{6,6,6},
    which is the Turán graph T(18, 3). This graph is K4-free.
    The largest triangle-free induced subgraph in K_{6,6,6} is formed by taking 
    all vertices from two of the three parts, resulting in a size of 6 + 6 = 12.
    Any sparser graph (subgraph of T(18,3)) will have a larger or equal 
    largest triangle-free induced subgraph because fewer edges make it easier 
    to find large triangle-free sets.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Vertices 0..17
    n = 18
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.append([u, v])
                    
    # Verify K4-free and triangle-free max size within time limit (optional safety check)
    # Given the deterministic nature and known properties, we can skip heavy verification
    # to ensure we return within time.
    
    return edges
