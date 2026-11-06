def construct():
    # Construct 8 disjoint copies of K4.
    # Total vertices n = 8 * 4 = 32.
    # This graph is C5-free because the maximum cycle length in K4 is 4,
    # and disjoint unions of C5-free graphs are C5-free.
    # The independence number of K4 is 1.
    # The independence number of the disjoint union is the sum of the independence numbers.
    # alpha = 8 * 1 = 8, which is < 9.
    # Thus, the graph is a valid witness for R(C5, K9) > 32.
    
    n = 32
    edges = []
    
    # Define the 8 blocks of 4 vertices
    for block_idx in range(8):
        base = block_idx * 4
        # Add all edges within the block (complete graph K4)
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
