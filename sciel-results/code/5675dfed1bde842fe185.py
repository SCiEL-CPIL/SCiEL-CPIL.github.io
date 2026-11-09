"""
Constructs a graph on 12 vertices (0..11) with minimum degree 3 and girth at least 4.
The objective is to minimize the number of distinct simple cycle lengths.

Analysis:
- The graph must be triangle-free (girth >= 4).
- It must contain at least one cycle.
- A graph with only one distinct cycle length (e.g., only 4-cycles) with minimum degree 3
  is impossible for finite simple graphs: a 3-regular graph with girth 4 must contain
  cycles of length 6 (e.g., K_{3,3} has 4- and 6-cycles; any 3-regular bipartite graph
  with girth 4 has cycles of length >= 4, and by the handshaking lemma on the cycle
  space, longer cycles necessarily appear).
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


def construct():
    # Deadline guard: 90 s wall-clock limit; use 85 s as safety margin.
    # For a static construction this is never reached, but the guard is kept
    # per the task contract.
    start_time = time.time()
    deadline = start_time + 85.0

    # Vertices 0-5: First K_{3,3}
    #   Partitions: A1 = {0, 1, 2}, B1 = {3, 4, 5}
    # Vertices 6-11: Second K_{3,3}
    #   Partitions: A2 = {6, 7, 8}, B2 = {9, 10, 11}

    edges = []

    # First K_{3,3}
    for u in (0, 1, 2):
        for v in (3, 4, 5):
            edges.append([u, v])

    if time.time() > deadline:
        return edges

    # Second K_{3,3}
    for u in (6, 7, 8):
        for v in (9, 10, 11):
            edges.append([u, v])

    # Ensure every element is a plain Python int (no numpy ints, no bools).
    final_edges = []
    for e in edges:
        u, v = e
        final_edges.append([int(u), int(v)])

    return final_edges
