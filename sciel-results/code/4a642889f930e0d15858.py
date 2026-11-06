def construct():
    # Construct 8 disjoint copies of K4.
    # Total vertices n = 8 * 4 = 32.
    # Each K4 has independence number 1.
    # Total independence number alpha = 8 * 1 = 8 < 9.
    # K4 contains no 5-cycles (max cycle length is 4).
    # Disjoint union of C5-free graphs is C5-free.
    
    n = 32
    edges = []
    num_blocks = 8
    block_size = 4
    
    for b in range(num_blocks):
        base = b * block_size
        # Generate all pairs (i, j) with i < j in the block
        for i in range(block_size):
            for j in range(i + 1, block_size):
                u = base + i
                v = base + j
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
