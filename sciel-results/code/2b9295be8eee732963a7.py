"""
Rewritten program for the suite task.
Maintains the optimal solution (two disjoint K_{3,3} components) which yields
a raw value of 2 (cycle lengths 4 and 6).
Includes a deadline guard as per directive.
"""

import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # The optimal known solution for minimizing distinct cycle lengths
    # with min degree 3 and triangle-free constraint is two disjoint K_{3,3} graphs.
    # K_{3,3} has cycle lengths 4 and 6.
    # Union of two such graphs has cycle lengths {4, 6}.
    # Raw value = 2.
    
    edges = []
    
    # Helper to add edges of K_{m,n} between two sets of vertices
    def add_k_mn(part_a, part_b):
        for u in part_a:
            for v in part_b:
                # Ensure unique representation (u < v)
                if u < v:
                    edges.append([u, v])
                else:
                    edges.append([v, u])
    
    # Component 1: Vertices 0-5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    add_k_mn(part_a1, part_b1)
    
    # Component 2: Vertices 6-11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    add_k_mn(part_a2, part_b2)
    
    # Check deadline (should not trigger for this static construction)
    if time.time() > deadline:
        # Fallback: return a valid simple graph if we ran out of time (unlikely)
        # For now, just return what we have, which is complete.
        pass
        
    # Deduplicate and sort for consistency
    unique_edges = set()
    for e in edges:
        u, v = e
        if u > v:
            u, v = v, u
        unique_edges.add((u, v))
        
    # Convert to list of lists of Python ints
    result = [[int(u), int(v)] for u, v in sorted(unique_edges)]
    
    return result
