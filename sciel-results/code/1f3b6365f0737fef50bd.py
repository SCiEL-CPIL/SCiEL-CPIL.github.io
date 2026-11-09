"""
Constructs a graph on 12 vertices (0..11) with minimum degree 3 and girth at least 4.
The objective is to minimize the number of distinct simple cycle lengths.

Analysis:
- The graph must be triangle-free (girth >= 4).
- It must contain at least one cycle.
- A graph with only one distinct cycle length (e.g., only 4-cycles) with minimum degree 3 is impossible for finite simple graphs due to structural constraints (leaves in the block-cut tree would have degree 2, or cycles of blocks create longer cycles).
- The minimum achievable number of distinct cycle lengths is 2.
- Two disjoint copies of K_{3,3} achieve this:
  - K_{3,3} has cycle lengths 4 and 6.
  - Disjoint union has cycle lengths {4, 6}.
  - Distinct lengths: 2.
  - Minimum degree: 3.
  - Girth: 4.
  - Contains cycles: Yes.

This solution returns the edges of two disjoint K_{3,3} components.
"""

import time
import sys

def construct():
    # Deadline guard: 90s wall-clock limit, use 80s as safety margin
    start_time = time.time()
    deadline = start_time + 80.0

    # Vertices 0-5: First K_{3,3}
    # Partitions: A1 = {0, 1, 2}, B1 = {3, 4, 5}
    edges = []
    
    # Generate edges for first K_{3,3}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([int(u), int(v)])
            
    # Check time
    if time.time() > deadline:
        return edges

    # Vertices 6-11: Second K_{3,3}
    # Partitions: A2 = {6, 7, 8}, B2 = {9, 10, 11}
    
    # Generate edges for second K_{3,3}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([int(u), int(v)])
            
    # Final type check and return
    # Ensure all elements are Python ints
    final_edges = []
    for e in edges:
        u, v = e
        if isinstance(u, int) and isinstance(v, int):
            final_edges.append([u, v])
        else:
            final_edges.append([int(u), int(v)])
            
    return final_edges
