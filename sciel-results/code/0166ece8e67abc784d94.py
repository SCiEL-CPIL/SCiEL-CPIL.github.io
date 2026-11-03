"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 6x6x6
    # Split into 2x2 blocks of size 3x3
    block_size = 3
    num_blocks = 2
    
    # 7 Strassen terms
    # Each term is a product of two (block_size x block_size) matrices
    # Rank of each term is block_size^3 = 27
    # Total rank = 7 * 27 = 189
    
    rank = 7 * (block_size ** 3)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for A blocks in X_s
    # s: term index 0..6
    # a, b: block row, col of A
    alpha = {}
    # s=0: A00 + A11
    alpha[(0, 0, 0)] = 1; alpha[(0, 1, 1)] = 1
    # s=1: A10 + A11
    alpha[(1, 1, 0)] = 1; alpha[(1, 1, 1)] = 1
    # s=2: A00
    alpha[(2, 0, 0)] = 1
    # s=3: A11
    alpha[(3, 1, 1)] = 1
    # s=4: A00 + A01
    alpha[(4, 0, 0)] = 1; alpha[(4, 0, 1)] = 1
    # s=5: A10 - A00
    alpha[(5, 1, 0)] = 1; alpha[(5, 0, 0)] = -1
    # s=6: A01 - A11
    alpha[(6, 0, 1)] = 1; alpha[(6, 1, 1)] = -1
    
    # Coefficients for B blocks in Y_s
    # s: term index 0..6
    # c, d: block row, col of B
    gamma = {}
    # s=0: B00 + B11
    gamma[(0, 0, 0)] = 1; gamma[(0, 1, 1)] = 1
    # s=1: B00
    gamma[(1, 0, 0)] = 1
    # s=2: B01 - B11
    gamma[(2, 0, 1)] = 1; gamma[(2, 1, 1)] = -1
    # s=3: B10 - B00
    gamma[(3, 1, 0)] = 1; gamma[(3, 0, 0)] = -1
    # s=4: B11
    gamma[(4, 1, 1)] = 1
    # s=5: B00 + B01
    gamma[(5, 0, 0)] = 1; gamma[(5, 0, 1)] = 1
    # s=6: B10 + B11
    gamma[(6, 1, 0)] = 1; gamma[(6, 1, 1)] = 1
    
    # Coefficients for M_s in C blocks
    # s: term index 0..6
    # i1, k1: block row, col of C
    beta = {}
    # s=0 (M1): C00 + C11
    beta[(0, 0, 0)] = 1; beta[(0, 1, 1)] = 1
    # s=1 (M2): C10 - C11
    beta[(1, 1, 0)] = 1; beta[(1, 1, 1)] = -1
    # s=2 (M3): C01 + C11
    beta[(2, 0, 1)] = 1; beta[(2, 1, 1)] = 1
    # s=3 (M4): C00 + C10
    beta[(3, 0, 0)] = 1; beta[(3, 1, 0)] = 1
    # s=4 (M5): -C00 + C01
    beta[(4, 0, 0)] = -1; beta[(4, 0, 1)] = 1
    # s=5 (M6): C11
    beta[(5, 1, 1)] = 1
    # s=6 (M7): C00
    beta[(6, 0, 0)] = 1
    
    # Construct U, V, W
    # r index: s * 27 + i0 * 9 + j0 * 3 + k0
    for s in range(7):
        for i0 in range(block_size):
            for j0 in range(block_size):
                for k0 in range(block_size):
                    r = s * (block_size ** 3) + i0 * (block_size ** 2) + j0 * block_size + k0
                    
                    # U: A part
                    for a in range(num_blocks):
                        for b in range(num_blocks):
                            coeff = alpha.get((s, a, b), 0)
                            if coeff != 0:
                                i = a * block_size + i0
                                j = b * block_size + j0
                                U[i * m + j][r] = coeff
                                
                    # V: B part
                    for c in range(num_blocks):
                        for d in range(num_blocks):
                            coeff = gamma.get((s, c, d), 0)
                            if coeff != 0:
                                j = c * block_size + j0
                                k = d * block_size + k0
                                V[j * p + k][r] = coeff
                                
                    # W: C part
                    for i1 in range(num_blocks):
                        for k1 in range(num_blocks):
                            coeff = beta.get((s, i1, k1), 0)
                            if coeff != 0:
                                i = i1 * block_size + i0
                                k = k1 * block_size + k0
                                W[k * n + i][r] = coeff
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
