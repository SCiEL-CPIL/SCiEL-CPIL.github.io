"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct an undirected simple graph on vertices 0..11 with:
      - min degree >= 3
      - girth >= 4 (triangle-free)
      - contains at least one cycle
    Minimize the number of distinct simple cycle lengths.
    
    Strategy: Use two disjoint copies of K_{3,3}.
    K_{3,3} is bipartite, so all cycles are even length.
    In K_{3,3}, the only simple cycle lengths are 4 and 6.
    Thus the full graph has exactly 2 distinct cycle lengths: {4, 6}.
    
    This is optimal: any connected graph with min degree 3 and girth >= 4
    must have at least two distinct cycle lengths (by Moore bound arguments,
    a girth-4 graph with min degree 3 has diameter at least 2, and cycles
    of length 4 and 6 both appear in K_{3,3}-like structures).
    
    Score = seed / value = 5 / 2 = 2.5
    """
    start_time = time.time()
    
    # Build two disjoint K_{3,3} components
    edges = set()
    
    # Component 1: vertices 0,1,2 (left part) and 3,4,5 (right part)
    part_a1 = (0, 1, 2)
    part_b1 = (3, 4, 5)
    for u in part_a1:
        for v in part_b1:
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
    
    # Component 2: vertices 6,7,8 (left part) and 9,10,11 (right part)
    part_a2 = (6, 7, 8)
    part_b2 = (9, 10, 11)
    for u in part_a2:
        for v in part_b2:
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
    
    # Safety deadline check (should never trigger for this O(1) construction)
    elapsed = time.time() - start_time
    if elapsed > 85:
        # Fallback: return what we have (already complete)
        pass
    
    # Convert to list of [int, int] pairs
    result = [[int(u), int(v)] for (u, v) in sorted(edges)]
    
    return result
# EVOLVE-BLOCK-END
