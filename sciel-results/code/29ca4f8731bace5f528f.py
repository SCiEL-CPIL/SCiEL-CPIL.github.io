"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 108 decomposition
    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10 + C11
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A00(B10-B11) -> C01 + C11
        U[i_base * m + j_base][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: A11(B01-B00) -> C00 + C10
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M5: (A00+A10)B11 -> C01 - C00
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + j_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        W[k_base * n + i_base][col] -= 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    # Part 1: 4x4x4 (i 0-3, j 0-3, k 0-3)
    # 8 blocks of 2x2x2
    for bi in range(2):
        for bj in range(2):
            # A block (bi, 0) * B block (0, bj)
            add_strassen_block(2*bi, 0, 2*bj)
            # A block (bi, 1) * B block (1, bj)
            add_strassen_block(2*bi, 2, 2*bj)
            
    # Part 2: 4x4x2 (i 0-3, j 0-3, k 4-5)
    # 4 blocks of 2x2x2
    for bi in range(2):
        # A block (bi, 0) * B block (0, k=4)
        add_strassen_block(2*bi, 0, 4)
        # A block (bi, 1) * B block (1, k=4)
        add_strassen_block(2*bi, 2, 4)
        
    # Part 3: 4x1x6 (i 0-3, j 4, k 0-5)
    # Standard multiplication
    for i in range(4):
        for k in range(6):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
