"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 36
    # 8 terms for k=0 (2x4x1)
    # 28 terms for k=1,2,3,4 (2x4x4) using 4 blocks of 2x2x2 (Strassen)
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: k=0, j=0..3, i=0..1
    # C_i0 = sum_j A_ij B_j0
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 0][col] = 1
            W[0 * n + i][col] = 1
            col += 1
            
    # Part 2: k=1..4
    # Split j into 0,1 and 2,3
    # Split k into 1,2 and 3,4
    # 4 blocks of 2x2x2
    
    def add_strassen_block(i_offset, j_offset, k_offset):
        nonlocal col
        # i is 0,1 relative to i_offset
        # j is 0,1 relative to j_offset
        # k is 0,1 relative to k_offset
        
        # M1 = (A00+A11)(B00+B11)
        # A00 is U[i_offset*2, j_offset*2], A11 is U[i_offset*2+1, j_offset*2+1]
        # B00 is V[j_offset*2, k_offset*2], B11 is V[j_offset*2+1, k_offset*2+1]
        
        # Term 1: (A00+A11)(B00+B11)
        U[i_offset * m + j_offset][col] += 1
        U[(i_offset + 1) * m + (j_offset + 1)][col] += 1
        V[j_offset * p + k_offset][col] += 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] += 1
        # W contributions: C00 += M1, C11 += M1
        W[k_offset * n + i_offset][col] += 1
        W[(k_offset + 1) * n + (i_offset + 1)][col] += 1
        col += 1
        
        # Term 2: (A10+A11)B00
        U[(i_offset + 1) * m + j_offset][col] += 1
        U[(i_offset + 1) * m + (j_offset + 1)][col] += 1
        V[j_offset * p + k_offset][col] += 1
        # W: C10 += M2
        W[k_offset * n + (i_offset + 1)][col] += 1
        col += 1
        
        # Term 3: A00(B10+B11)
        U[i_offset * m + j_offset][col] += 1
        V[(j_offset + 1) * p + k_offset][col] += 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] += 1
        # W: C01 += M3
        W[(k_offset + 1) * n + i_offset][col] += 1
        col += 1
        
        # Term 4: A11(B01-B10)
        U[(i_offset + 1) * m + (j_offset + 1)][col] += 1
        V[j_offset * p + (k_offset + 1)][col] += 1
        V[(j_offset + 1) * p + k_offset][col] += -1
        # W: C00 += M4
        W[k_offset * n + i_offset][col] += 1
        col += 1
        
        # Term 5: (A00+A10)B11
        U[i_offset * m + j_offset][col] += 1
        U[(i_offset + 1) * m + j_offset][col] += 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] += 1
        # W: C00 -= M5, C10 += M5
        W[k_offset * n + i_offset][col] += -1
        W[k_offset * n + (i_offset + 1)][col] += 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B01)
        U[(i_offset + 1) * m + j_offset][col] += 1
        U[i_offset * m + j_offset][col] += -1
        V[j_offset * p + k_offset][col] += 1
        V[j_offset * p + (k_offset + 1)][col] += 1
        # W: C11 += M6
        W[(k_offset + 1) * n + (i_offset + 1)][col] += 1
        col += 1
        
        # Term 7: (A01-A11)(B10+B11)
        U[i_offset * m + (j_offset + 1)][col] += 1
        U[(i_offset + 1) * m + (j_offset + 1)][col] += -1
        V[(j_offset + 1) * p + k_offset][col] += 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] += 1
        # W: C00 -= M7, C01 += M7
        W[k_offset * n + i_offset][col] += -1
        W[(k_offset + 1) * n + i_offset][col] += 1
        col += 1

    # Block 1: i=0,1; j=0,1; k=1,2
    add_strassen_block(0, 0, 1)
    # Block 2: i=0,1; j=0,1; k=3,4
    add_strassen_block(0, 0, 3)
    # Block 3: i=0,1; j=2,3; k=1,2
    add_strassen_block(0, 2, 1)
    # Block 4: i=0,1; j=2,3; k=3,4
    add_strassen_block(0, 2, 3)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
