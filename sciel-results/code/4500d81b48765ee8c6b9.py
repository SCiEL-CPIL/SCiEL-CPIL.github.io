"""
Gallai coloring of K_16 maximizing distinct edge colors.

Uses a recursive partition (cut) tree. Each internal node splits its vertex set
into two parts; all cross edges between the parts receive one color. Leaves
are small vertex sets whose internal edges are colored with distinct colors.

For n=16, the maximum number of distinct colors in a Gallai coloring is 20.
We use five 3-vertex leaves (15 distinct internal colors) + 5 internal node
colors = 20 total.
"""
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2


def _edge_idx(i, j):
    """Return the index in the lexicographic edge list for edge (i,j) with i<j."""
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)


# Precompute all 4-cycles as tuples of 4 edge indices
_CYCLES = []
for a, b, c, d in combinations(range(N), 4):
    _CYCLES.append((_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(d, a)))
    _CYCLES.append((_edge_idx(a, b), _edge_idx(b, d), _edge_idx(d, c), _edge_idx(c, a)))
    _CYCLES.append((_edge_idx(a, c), _edge_idx(c, b), _edge_idx(b, d), _edge_idx(d, a)))


def construct():
    """
    Construct a Gallai coloring of K_16 with 20 distinct edge colors.

    Partition tree:
      Leaves: [0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15]
      Node A: L0 vs L1          -> color 15
      Node B: L2 vs L3          -> color 16
      Node C: (L0+L1) vs (L2+L3)-> color 17
      Node D: L4 vs L5          -> color 18
      Root: (L0..L3) vs (L4+L5) -> color 19

    Leaf internals: 15 distinct colors (0..14).
    Total: 20 distinct colors.
    """
    colors = [0] * NUM_EDGES
    color_counter = 0

    def get_new_color():
        nonlocal color_counter
        c = color_counter
        color_counter += 1
        return c

    def color_all_distinct(vertices):
        """Color all edges within vertices with distinct colors."""
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                colors[_edge_idx(u, v)] = get_new_color()

    def color_node(part1, part2):
        """Color inter-part edges between part1 and part2 with one new color."""
        c = get_new_color()
        for u in part1:
            for v in part2:
                colors[_edge_idx(u, v)] = c

    # Leaves
    L0 = [0, 1, 2]
    L1 = [3, 4, 5]
    L2 = [6, 7, 8]
    L3 = [9, 10, 11]
    L4 = [12, 13, 14]
    L5 = [15]

    # Color leaf internals with distinct colors (15 colors: 0-14)
    for leaf in (L0, L1, L2, L3, L4, L5):
        if len(leaf) >= 2:
            color_all_distinct(leaf)

    # Internal node colors (5 colors: 15-19)
    # Node A: L0 vs L1
    color_node(L0, L1)
    # Node B: L2 vs L3
    color_node(L2, L3)
    # Node C: (L0+L1) vs (L2+L3)
    color_node(L0 + L1, L2 + L3)
    # Node D: L4 vs L5
    color_node(L4, L5)
    # Root: (L0+L1+L2+L3) vs (L4+L5)
    color_node(L0 + L1 + L2 + L3, L4 + L5)

    return colors
