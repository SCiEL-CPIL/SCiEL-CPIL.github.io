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

def construct():
    # We construct two disjoint K_{3,3} graphs.
    # Vertices 0-5 form the first K_{3,3}: A={0,1,2}, B={3,4,5}
    # Vertices 6-11 form the second K_{3,3}: A={6,7,8}, B={9,10,11}

    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]

    edges = []

    # Generate edges for first K_{3,3} component
    for u in part_a1:
        for v in part_b1:
            edges.append([u, v])

    # Generate edges for second K_{3,3} component
    for u in part_a2:
        for v in part_b2:
            edges.append([u, v])

    # Validation and cleanup: remove loops and duplicates, normalize ordering
    final_edges = []
    seen_pairs = set()

    for e in edges:
        u, v = e
        if u == v:
            continue
        pair_key = (min(u, v), max(u, v))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        final_edges.append([u, v])

    # Sort for deterministic output
    final_edges.sort()

    # Strict type enforcement: ensure every element is a Python int
    # (rejecting NumPy integers and bools)
    validated_edges = []
    for e in final_edges:
        u, v = e
        if not isinstance(u, int) or isinstance(u, bool):
            u = int(u)
        if not isinstance(v, int) or isinstance(v, bool):
            v = int(v)
        validated_edges.append([u, v])

    return validated_edges
