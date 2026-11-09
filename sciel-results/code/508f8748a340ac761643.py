"""
Constructs a diameter-critical graph with maximum edges.
The optimal graph for n=16 is K_{8,8} (complete bipartite graph).
It has 64 edges.
Diameter is 2.
Removing any edge increases the diameter to 3 (endpoints become distance 3 apart).
"""

import time

def construct():
    """
    Returns the edge list of K_{8,8}.
    """
    # Deadline guard (not strictly necessary for this O(n^2) construction, 
    # but included per directive for safety in search contexts)
    start_time = time.time()
    deadline = start_time + 60.0
    
    n = 16
    mid = n // 2
    
    edges = []
    # Generate all edges between {0, ..., mid-1} and {mid, ..., n-1}
    for u in range(mid):
        for v in range(mid, n):
            # Check deadline just in case (should never trigger)
            if time.time() > deadline:
                return edges
            edges.append([u, v])
            
    return edges
