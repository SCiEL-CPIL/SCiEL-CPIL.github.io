"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    """
    Construct a triangle-free graph on vertices 0..11 with min degree >= 3
    that minimizes the number of distinct simple cycle lengths.
    
    Strategy: Use two disjoint K_{3,3} components.
    - K_{3,3} is bipartite, so all cycles have even length.
    - In K_{3,3}, the only possible cycle lengths are 4 and 6.
    - Both 4-cycles and 6-cycles exist in K_{3,3}.
    - Therefore, the set of cycle lengths is {4, 6}, giving 2 distinct lengths.
    
    This is optimal because:
    - A graph with minimum degree 3 must contain a cycle.
    - If triangle-free, the shortest cycle is at least 4.
    - Having only 1 distinct cycle length L would require all cycles to have length L.
    - For a 3-regular graph, it's difficult/impossible to have all cycles the same length
      while maintaining the degree constraint, especially for small n.
    - The value 2 is the known minimum for this constraint set.
    """
    edges = []
    
    # Component 1: K_{3,3} on vertices 0,1,2 (Left) and 3,4,5 (Right)
    # Left partition: {0, 1, 2}
    # Right partition: {3, 4, 5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
    
    # Component 2: K_{3,3} on vertices 6,7,8 (Left) and 9,10,11 (Right)
    # Left partition: {6, 7, 8}
    # Right partition: {9, 10, 11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
    
    # Ensure all types are Python int and list of lists
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result

# EVOLVE-BLOCK-END
