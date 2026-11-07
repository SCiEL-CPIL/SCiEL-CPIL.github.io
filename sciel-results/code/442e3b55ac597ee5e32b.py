"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs an undirected simple graph on vertices 0..11.
    Constraints:
    - Every vertex degree >= 3
    - Triangle-free (girth >= 4)
    - Contains a cycle
    - Minimize number of distinct simple cycle lengths
    
    Strategy:
    The theoretical minimum for the number of distinct cycle lengths in a 
    simple graph with minimum degree 3 is 2. 
    This is achieved by the complete bipartite graph K_{3,3}, which has 
    cycles of length 4 and 6.
    Since we have 12 vertices, we can use two disjoint K_{3,3} components.
    This results in distinct cycle lengths {4, 6}, so the raw value is 2.
    """
    
    n = 12
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (part A) and 3,4,5 (part B)
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6,7,8 (part A) and 9,10,11 (part B)
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
            
    # Validation (optional but good for safety)
    # Check degrees
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d < 3 for d in deg):
        # Fallback to a known valid structure if this somehow fails (it won't)
        # But K_{3,3} always has degree 3.
        pass
        
    # Check triangle-free
    # K_{3,3} is bipartite, so it is triangle-free.
    
    # Check cycle
    # K_{3,3} contains C4.
    
    # Ensure types are correct
    result = []
    for u, v in edges:
        result.append([int(u), int(v)])
        
    return result
# EVOLVE-BLOCK-END
