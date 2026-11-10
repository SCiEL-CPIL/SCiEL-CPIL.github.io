"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a graph on vertices 0..11 with:
    - Every vertex has degree at least 3
    - Every simple cycle has length at least 4 (triangle-free)
    - Minimize the number of distinct simple cycle lengths
    
    Strategy: Use two disjoint copies of K_{3,3}.
    K_{3,3} is 3-regular, bipartite (so girth 4, no triangles),
    and contains only cycles of length 4 and 6.
    Two disjoint copies give 12 vertices, all degree 3, girth 4,
    and only cycle lengths {4, 6} -> raw value = 2.
    
    This is optimal for 3-regular graphs on 12 vertices as 1 distinct 
    cycle length is not achievable (no 3-regular graph on 12 vertices 
    has all cycles of the same length, e.g., all 4s or all 5s).
    """
    start_time = time.time()
    
    # First copy of K_{3,3}: vertices 0-5
    # Left partition: {0, 1, 2}, Right partition: {3, 4, 5}
    edges = set()
    for i in range(3):
        for j in range(3):
            u = i
            v = 3 + j
            edges.add((u, v))
            
    # Second copy of K_{3,3}: vertices 6-11
    # Left partition: {6, 7, 8}, Right partition: {9, 10, 11}
    for i in range(3):
        for j in range(3):
            u = 6 + i
            v = 9 + j
            edges.add((u, v))
    
    # Convert to list of [u, v] lists with sorted endpoints
    result = []
    for u, v in sorted(edges):
        if u > v:
            u, v = v, u
        result.append([u, v])
        
    # Explicit deadline guard (though this is instantaneous)
    if time.time() - start_time > 50:
        # Fallback is the same, so just return
        pass
        
    return result

# EVOLVE-BLOCK-END
