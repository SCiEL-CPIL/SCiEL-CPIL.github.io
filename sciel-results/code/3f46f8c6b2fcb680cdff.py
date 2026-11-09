"""
Constructs a graph on vertices 0..11 with minimum degree 3, girth at least 4,
and exactly 2 distinct simple cycle lengths (4 and 6).

The graph consists of two disjoint copies of K_{3,3}.
- K_{3,3} is bipartite, so it is triangle-free (girth >= 4).
- The simple cycles in K_{3,3} have lengths 4 and 6.
- Since the components are disjoint, the set of cycle lengths for the whole graph is {4, 6}.
- The number of distinct cycle lengths is 2.
- It is a known lower bound that min-degree 3 triangle-free graphs cannot have only 1 distinct cycle length.
"""

import time

_START_TIME = None

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Safety deadline: 85 seconds to leave margin for evaluator overhead
    DEADLINE = _START_TIME + 85
    
    # We construct two disjoint K_{3,3} graphs.
    # Vertices 0-5 form the first K_{3,3}
    # Vertices 6-11 form the second K_{3,3}
    
    # Part A indices for first component
    part_a1 = [0, 1, 2]
    # Part B indices for first component
    part_b1 = [3, 4, 5]
    
    # Part A indices for second component
    part_a2 = [6, 7, 8]
    # Part B indices for second component
    part_b2 = [9, 10, 11]
    
    edges = []
    
    # Generate edges for first component
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])
    
    # Generate edges for second component
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])
    
    # Validation and cleanup
    final_edges = []
    seen_pairs = set()
    
    for e in edges:
        u, v = e
        # Convert to int to ensure type correctness
        u_int = int(u)
        v_int = int(v)
        
        if u_int == v_int:
            continue
            
        # Normalize order for duplicate checking
        pair_key = (min(u_int, v_int), max(u_int, v_int))
        
        if pair_key in seen_pairs:
            continue
            
        seen_pairs.add(pair_key)
        final_edges.append([u_int, v_int])
        
    # Sort for deterministic output
    final_edges.sort()
    
    if time.time() > DEADLINE:
        return final_edges
        
    return final_edges
