"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    """
    Constructs a graph on vertices 0..11.
    Uses two disjoint K3,3 components.
    K3,3 is bipartite, girth 4.
    Cycle lengths in K3,3 are 4 and 6.
    Distinct lengths: {4, 6}. Count: 2.
    Min degree: 3.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 12
    edges = []
    
    # Component 1: K3,3 on vertices 0-5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # Component 2: K3,3 on vertices 6-11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    # Check time (not really needed for this fast construction, but good practice)
    if time.time() > deadline:
        # This should never happen, but if it does, return what we have
        pass
        
    # Ensure no duplicates or loops (K3,3 construction is clean)
    # Convert to sorted tuple for uniqueness check if needed, but we know it's unique
    unique_edges = []
    seen = set()
    for e in edges:
        u, v = e
        if u == v:
            continue
        key = tuple(sorted((u, v)))
        if key not in seen:
            seen.add(key)
            unique_edges.append(list(key))
            
    return unique_edges
# EVOLVE-BLOCK-END
