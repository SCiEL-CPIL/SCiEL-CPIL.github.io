import time
from itertools import product

def construct():
    """
    Returns edges for a graph on 12 vertices (0..11) with:
      - minimum degree >= 3
      - girth >= 4 (triangle-free)
      - at least one simple cycle
    minimizing the number of distinct simple cycle lengths.

    Strategy: keep the proven two-disjoint-K3,3 design (cycle lengths {4,6},
    distinct count = 2, score 2.5). Within this family the theoretical lower
    bound is 1 (a single cycle length), but a 3-regular triangle-free graph on
    12 vertices cannot have all cycles of one length while remaining
    bipartite with min-degree 3 (K3,3 forces both 4-cycles and 6-cycles; the
    Petersen graph is not 3-regular bipartite on 12 vertices; other small
    3-regular triangle-free graphs introduce additional cycle lengths).
    Hence we retain the family-best construction and add a lightweight
    deadline guard.
    """
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin well inside the 90 s wall-clock limit

    edges = []
    # Component 1: K_{3,3} on vertices {0,1,2} x {3,4,5}
    for a, b in product((0, 1, 2), (3, 4, 5)):
        edges.append([int(a), int(b)])

    # Component 2: K_{3,3} on vertices {6,7,8} x {9,10,11}
    for a, b in product((6, 7, 8), (9, 10, 11)):
        edges.append([int(a), int(b)])

    # Deadline guard (should never trigger for this O(1) construction)
    if time.time() > deadline:
        return edges

    return edges


if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
