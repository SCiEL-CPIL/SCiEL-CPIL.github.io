"""
Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_3,3 components.
Each K_3,3 has minimum degree 3 and girth 4.
The cycle lengths in K_3,3 are exactly 4 and 6.
Thus, the whole graph has distinct cycle lengths {4, 6}, count = 2.
This is the theoretical minimum for any graph with minimum degree >= 3.
"""
import time

def construct():
    """
    Constructs a graph on 12 vertices (0..11) consisting of two disjoint K_3,3 components.
    Ensures all vertices are Python int, edges are lists of Python ints.
    Returns a list of edges.
    """
    start_time = time.time()
    # Safety margin: The task allows 90s, we use 85s as a hard stop.
    deadline = start_time + 85.0

    edges = []

    # Component 1: K_3,3 on vertices 0, 1, 2 (left) and 3, 4, 5 (right)
    part1_left = [0, 1, 2]
    part1_right = [3, 4, 5]
    for u in part1_left:
        for v in part1_right:
            # Ensure strict Python int type
            edges.append([int(u), int(v)])

    # Check deadline (extremely unlikely to be hit for static construction)
    if time.time() > deadline:
        return edges

    # Component 2: K_3,3 on vertices 6, 7, 8 (left) and 9, 10, 11 (right)
    part2_left = [6, 7, 8]
    part2_right = [9, 10, 11]
    for u in part2_left:
        for v in part2_right:
            # Ensure strict Python int type
            edges.append([int(u), int(v)])

    # Final check before return
    if time.time() > deadline:
        return edges

    # Optional: Lightweight validation to ensure no accidental duplicates or loops
    # This is O(E^2) which is 81^2 ~ 6561 ops, negligible.
    # However, since the construction is by disjoint sets, it's safe.
    # We skip heavy validation to save time, but ensure types are correct.
    
    return edges
