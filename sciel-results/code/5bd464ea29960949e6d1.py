    n, m, p = DIMENSIONS
    # rank calculation
    # blocks of size 2
    # 6/2 = 3 blocks per dim
    blocks_n = n // 2
    blocks_m = m // 2
    blocks_p = p // 2
    # For 6x6x6, blocks are 3x3x3
    # Rank per 2x2x2 block is 7 (Strassen)
    # Total rank = blocks_n * blocks_m * blocks_p * 7 ?
    # Wait, the block decomposition logic in the code:
    # "There are 3*3*3 = 27 triplets."
    # Triplet (I, J, K) where I is block row of A, J is block col of A / row of B, K is block col of B.
    # I in 0..2, J in 0..2, K in 0..2.
    # This corresponds to blocks_n=3, blocks_m=3, blocks_p=3.
    # So rank = 3 * 3 * 3 * 7 = 189.
    
    rank = blocks_n * blocks_m * blocks_p * 7
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
