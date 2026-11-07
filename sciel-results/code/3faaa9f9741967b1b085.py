"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 12, 'minimum_degree': 3, 'girth': 4}

def construct():
    """
    Construct a graph on vertices 0..11 with:
    - Every vertex has degree at least 3
    - Every simple cycle has length at least 4 (triangle-free)
    - Minimize the number of distinct simple cycle lengths
    
    Strategy: Two disjoint K_{3,3} components.
    - Bipartite, so no odd cycles (girth >= 4 guaranteed)
    - Minimum degree is 3 (satisfies degree >= 3)
    - K_{3,3} has exactly cycles of length 4 and 6
    - Result: 2 distinct cycle lengths {4, 6}
    
    This is optimal: 1 distinct cycle length is not achievable with min degree 3
    on 12 vertices (proven by analysis of C4-only and C6-only graph constraints).
    """
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    
    edges = []
    
    # First K_{3,3}: vertices 0,1,2 in part A, 3,4,5 in part B
    for i in range(3):
        for j in range(3):
            edges.append([i, 3 + j])
    
    # Second K_{3,3}: vertices 6,7,8 in part A, 9,10,11 in part B
    for i in range(3):
        for j in range(3):
            edges.append([6 + i, 9 + j])
    
    # Ensure all types are correct: Python int, list of lists
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
