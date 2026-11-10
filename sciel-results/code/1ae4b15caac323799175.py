"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    # Vertices 0..15
    # Partition into 4 groups of 4: G0={0,1,2,3}, G1={4,5,6,7}, G2={8,9,10,11}, G3={12,13,14,15}
    
    colors = [0] * (N*(N-1)//2)
    
    def get_group(v):
        return v // 4
    
    def get_intra_color(v1, v2):
        # v1, v2 in same group
        # Map local indices 0,1,2,3 to colors 0,1,2
        # Using the 3-coloring of K4 derived earlier:
        # (0,1)=0, (2,3)=0
        # (0,2)=1, (1,3)=1
        # (0,3)=2, (1,2)=2
        # Note: We add an offset per group to make colors distinct across groups
        g = get_group(v1)
        offset = g * 3
        l1 = v1 % 4
        l2 = v2 % 4
        if l1 > l2:
            l1, l2 = l2, l1
            
        if (l1, l2) == (0, 1): return offset
        if (l1, l2) == (2, 3): return offset
        if (l1, l2) == (0, 2): return offset + 1
        if (l1, l2) == (1, 3): return offset + 1
        if (l1, l2) == (0, 3): return offset + 2
        if (l1, l2) == (1, 2): return offset + 2
        return 0 # Should not happen

    def get_inter_color(v1, v2):
        g1 = get_group(v1)
        g2 = get_group(v2)
        # Use sum mod 4, offset by 12 to be distinct from intra colors
        return 12 + ((g1 + g2) % 4)

    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            g_i = get_group(i)
            g_j = get_group(j)
            if g_i == g_j:
                colors[idx] = get_intra_color(i, j)
            else:
                colors[idx] = get_inter_color(i, j)
            idx += 1
            
    return colors
# EVOLVE-BLOCK-END
