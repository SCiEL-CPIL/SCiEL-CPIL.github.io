"""Rewritten program: two disjoint K_{3,3} components."""
# EVOLVE-BLOCK-START
import time

def construct():
    """Return edges of two disjoint K_{3,3} graphs on vertices 0..11.
    
    Component 1: K_{3,3} with partitions {0,1,2} and {3,4,5}
    Component 2: K_{3,3} with partitions {6,7,8} and {9,10,11}
    
    This graph is:
    - Triangle-free (bipartite)
    - 3-regular (every vertex has degree 3)
    - Contains cycles (K_{3,3} has cycles of length 4 and 6)
    - Has exactly 2 distinct cycle lengths (4 and 6)
    """
    start_time = time.time()
    deadline = start_time + 59.0  # Safety margin
    
    edges = set()
    
    # First K_{3,3}: partitions {0,1,2} x {3,4,5}
    for u in range(3):
        if time.time() > deadline:
            break
        for v in range(3, 6):
            edges.add(tuple(sorted((u, v))))
    
    # Second K_{3,3}: partitions {6,7,8} x {9,10,11}
    for u in range(6, 9):
        if time.time() > deadline:
            break
        for v in range(9, 12):
            edges.add(tuple(sorted((u, v))))
            
    # Fallback if deadline exceeded (should not happen for this size)
    if time.time() > deadline:
        # Return whatever we have, or a minimal valid graph if empty
        if not edges:
             # Minimal 3-regular triangle-free graph with cycles?
             # Just return the first K33 if we managed to build it
             pass 
             
    return [list(e) for e in sorted(edges)]
# EVOLVE-BLOCK-END
