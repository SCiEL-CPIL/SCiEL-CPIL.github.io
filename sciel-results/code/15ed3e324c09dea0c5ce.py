"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """
    Constructs an undirected simple graph on vertices 0..11.
    Constraints:
    - Min degree >= 3
    - Triangle-free (girth >= 4)
    - Contains a cycle
    - Minimize distinct cycle lengths
    
    Strategy:
    Use two disjoint copies of K_{3,3}.
    K_{3,3} is bipartite (triangle-free).
    K_{3,3} has cycles of length 4 and 6 only.
    Disconnected components do not create new cycle lengths.
    Total distinct cycle lengths: {4, 6} -> count = 2.
    """
    start_time = time.time()
    deadline = start_time + 55.0

    edges = []

    # Component 1: K_{3,3} on vertices 0..5
    # Partitions: {0, 1, 2} and {3, 4, 5}
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])

    # Component 2: K_{3,3} on vertices 6..11
    # Partitions: {6, 7, 8} and {9, 10, 11}
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])

    # Sort edges for determinism
    edges.sort()

    # Ensure types are strictly Python int and list
    # Although u and v are already ints from the lists above, 
    # we enforce this to be safe against any unexpected type coercion.
    result = []
    for e in edges:
        u = int(e[0])
        v = int(e[1])
        if isinstance(u, bool) or isinstance(v, bool):
            # bool is subclass of int, so cast to int explicitly to avoid bools
            u = int(u)
            v = int(v)
        result.append([u, v])

    # Deadline guard
    if time.time() > deadline:
        return result

    return result

# EVOLVE-BLOCK-END
