import time

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Define groups of 4 vertices
    groups = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    # Internal color patterns for each group
    # Pattern uses 4 distinct colors for the 6 edges of K4
    # Vertices in group: 0, 1, 2, 3 (local indices)
    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Colors: c0, c1, c2, c3
    # Mapping:
    # (0,1) -> c0
    # (2,3) -> c0
    # (0,2) -> c1
    # (1,3) -> c1
    # (0,3) -> c2
    # (1,2) -> c3
    
    # Group 0 uses colors 0, 1, 2, 3
    # Group 1 uses colors 4, 5, 6, 7
    # Group 2 uses colors 8, 9, 10, 11
    # Group 3 uses colors 12, 13, 14, 15
    
    internal_colors = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    # Inter-group colors
    # Z1: between G0 and G1
    # Z2: between G2 and G3
    # Z3: between {G0, G1} and {G2, G3}
    
    Z1 = 16
    Z2 = 17
    Z3 = 18
    
    # Initialize colors array
    colors = [0] * (N * (N - 1) // 2)
    
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        # Index for edge (i, j) in lexicographic order of pairs (i, j) with 0 <= i < j < N
        # Count edges before i: i * (N - 1) - i * (i - 1) // 2
        # Edges from i to j: j - i - 1
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Assign internal colors
    for g_idx, group in enumerate(groups):
        c0, c1, c2, c3 = internal_colors[g_idx]
        v0, v1, v2, v3 = group
        
        # (v0, v1) -> c0
        colors[get_idx(v0, v1)] = c0
        # (v2, v3) -> c0
        colors[get_idx(v2, v3)] = c0
        # (v0, v2) -> c1
        colors[get_idx(v0, v2)] = c1
        # (v1, v3) -> c1
        colors[get_idx(v1, v3)] = c1
        # (v0, v3) -> c2
        colors[get_idx(v0, v3)] = c2
        # (v1, v2) -> c3
        colors[get_idx(v1, v2)] = c3
        
    # Assign inter-group colors
    # Helper to assign color to all edges between two groups
    def assign_inter(g1_idx, g2_idx, color_val):
        for u in groups[g1_idx]:
            for v in groups[g2_idx]:
                colors[get_idx(u, v)] = color_val
                
    # G0-G1: Z1
    assign_inter(0, 1, Z1)
    # G2-G3: Z2
    assign_inter(2, 3, Z2)
    # G0-G2: Z3
    assign_inter(0, 2, Z3)
    # G0-G3: Z3
    assign_inter(0, 3, Z3)
    # G1-G2: Z3
    assign_inter(1, 2, Z3)
    # G1-G3: Z3
    assign_inter(1, 3, Z3)
    
    return colors
# EVOLVE-BLOCK-END
