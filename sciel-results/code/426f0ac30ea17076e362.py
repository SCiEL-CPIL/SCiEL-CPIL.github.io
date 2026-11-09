"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a Gallai coloring of K_16 maximizing distinct edge colors.
    Uses recursive partition: split into parts, color inter-part edges monochromatically,
    recurse on each part. Leaves of size <= 3 get all-distinct edge colors.
    Maximum distinct colors = 20 (five 3-vertex leaves + one 1-vertex leaf).
    """
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s limit
    
    # Precompute edge index mapping
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[(i, j)] = idx
            idx += 1
    
    colors = [0] * (N*(N-1)//2)
    color_counter = [0]  # mutable counter for distinct colors used
    
    def get_new_color():
        c = color_counter[0]
        color_counter[0] += 1
        return c
    
    def color_all_distinct(vertices):
        """Color all edges within vertices with distinct colors."""
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if u > v:
                    u, v = v, u
                colors[edge_idx[(u, v)]] = get_new_color()
    
    def color_node(part1, part2):
        """Color inter-part edges between part1 and part2 with one color."""
        c = get_new_color()
        for u in part1:
            for v in part2:
                a, b = min(u, v), max(u, v)
                colors[edge_idx[(a, b)]] = c
    
    # Fixed partition approach achieving 20 distinct colors
    # Leaves: 5 groups of 3 + 1 single vertex
    leaves = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
        [15]
    ]
    
    # Binary tree over 6 leaves:
    #   Node0: leaves[0] + leaves[1]  -> [0..5]
    #   Node1: leaves[2] + leaves[3]  -> [6..11]
    #   Node2: Node0 + Node1          -> [0..11]
    #   Node3: leaves[4] + leaves[5]  -> [12..15]
    #   Root: Node2 + Node3           -> [0..15]
    
    # Color leaf internals (distinct colors within each leaf of size 3)
    for leaf in leaves:
        if len(leaf) >= 2:
            color_all_distinct(leaf)
    
    # Color tree cross-edges
    # Node0: leaves[0] vs leaves[1]
    color_node(leaves[0], leaves[1])
    # Node1: leaves[2] vs leaves[3]
    color_node(leaves[2], leaves[3])
    # Node2: (leaves[0]+leaves[1]) vs (leaves[2]+leaves[3])
    part2a = leaves[0] + leaves[1]
    part2b = leaves[2] + leaves[3]
    color_node(part2a, part2b)
    # Node3: leaves[4] vs leaves[5]
    color_node(leaves[4], leaves[5])
    # Root: (leaves[0..3]) vs (leaves[4]+leaves[5])
    part_root_a = leaves[0] + leaves[1] + leaves[2] + leaves[3]
    part_root_b = leaves[4] + leaves[5]
    color_node(part_root_a, part_root_b)
    
    # Verify: count distinct colors
    distinct_colors = len(set(colors))
    
    # Verify: no rainbow 4-cycle
    # For each 4-vertex subset, check all 3 cycles
    from itertools import combinations
    
    def check_no_rainbow_4cycle(colors, edge_idx, N):
        for quad in combinations(range(N), 4):
            a, b, c, d = quad
            # Three 4-cycles on {a,b,c,d}:
            # Cycle 1: a-b-c-d-a (edges ab, bc, cd, da)
            # Cycle 2: a-b-d-c-a (edges ab, bd, dc, ca)
            # Cycle 3: a-c-b-d-a (edges ac, cb, bd, da)
            cycles = [
                [(a,b), (b,c), (c,d), (d,a)],
                [(a,b), (b,d), (d,c), (c,a)],
                [(a,c), (c,b), (b,d), (d,a)]
            ]
            for cycle in cycles:
                cycle_colors = set()
                for (u, v) in cycle:
                    if u > v:
                        u, v = v, u
                    cycle_colors.add(colors[edge_idx[(u, v)]])
                if len(cycle_colors) == 4:
                    return False  # rainbow 4-cycle found
        return True
    
    if not check_no_rainbow_4cycle(colors, edge_idx, N):
        # This should never happen with our construction, but safety check
        return colors
    
    return colors

# EVOLVE-BLOCK-END
