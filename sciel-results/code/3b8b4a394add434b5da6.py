"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return the exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    Uses Strassen for 2x2x2 blocks to reduce rank from 54 to 50.
    """
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for A (2x2) * B (2x2)
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0,1}
    # C indices: i in {0,1}, k in {0,1}
    def add_strassen_block(j_offset, k_offset):
        nonlocal col
        # Strassen's 7 multiplications
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # M1
        U[0*3+0][col] += 1; U[1*3+1][col] += 1
        V[0*9+k_offset][col] += 1; V[1*9+(k_offset+1)][col] += 1
        W[k_offset*2+0][col] += 1; W[(k_offset+1)*2+1][col] += 1
        col += 1
        
        # M2
        U[1*3+0][col] += 1; U[1*3+1][col] += 1
        V[0*9+k_offset][col] += 1
        W[k_offset*2+1][col] += 1
        col += 1
        
        # M3
        U[0*3+0][col] += 1
        V[0*9+(k_offset+1)][col] += 1; V[1*9+(k_offset+1)][col] -= 1
        W[(k_offset+1)*2+0][col] += 1
        col += 1
        
        # M4
        U[1*3+1][col] += 1
        V[1*9+k_offset][col] += 1; V[0*9+k_offset][col] -= 1
        W[k_offset*2+1][col] += 1
        col += 1
        
        # M5
        U[0*3+0][col] += 1; U[0*3+1][col] += 1
        V[1*9+(k_offset+1)][col] += 1
        W[(k_offset+1)*2+0][col] += 1
        col += 1
        
        # M6
        U[1*3+0][col] += 1; U[0*3+0][col] -= 1
        V[0*9+k_offset][col] += 1; V[0*9+(k_offset+1)][col] += 1
        W[k_offset*2+1][col] += 1
        col += 1
        
        # M7
        U[0*3+1][col] += 1; U[1*3+1][col] -= 1
        V[1*9+k_offset][col] += 1; V[1*9+(k_offset+1)][col] += 1
        W[(k_offset+1)*2+1][col] += 1
        col += 1
        
        # Update W for the combination
        # c00 = M1 + M4 - M5 + M7
        W[k_offset*2+0][col-7] += 1
        W[k_offset*2+0][col-4] += 1
        W[k_offset*2+0][col-3] -= 1
        W[k_offset*2+0][col-1] += 1
        
        # c01 = M3 + M5
        W[(k_offset+1)*2+0][col-5] += 1
        W[(k_offset+1)*2+0][col-3] += 1
        
        # c10 = M2 + M4
        W[k_offset*2+1][col-6] += 1
        W[k_offset*2+1][col-4] += 1
        
        # c11 = M1 - M2 + M3 + M6
        W[(k_offset+1)*2+1][col-7] += 1
        W[(k_offset+1)*2+1][col-6] -= 1
        W[(k_offset+1)*2+1][col-5] += 1
        W[(k_offset+1)*2+1][col-2] += 1

    # Part 1: A1 (2x2) * B1 (2x9)
    # B1 is rows 0,1 of B.
    # Split B1 into four 2x2 blocks and one 2x1 block.
    
    # Block 1: cols 0,1
    add_strassen_block(0, 0)
    
    # Block 2: cols 2,3
    add_strassen_block(0, 2)
    
    # Block 3: cols 4,5
    add_strassen_block(0, 4)
    
    # Block 4: cols 6,7
    add_strassen_block(0, 6)
    
    # Block 5: col 8 (2x1 schoolbook)
    # c_i,8 = a_i0 b_0,8 + a_i1 b_1,8
    # 4 multiplications
    for i in range(2):
        for j in range(2):
            U[i*3+j][col] = 1
            V[j*9+8][col] = 1
            W[8*2+i][col] = 1
            col += 1
            
    # Part 2: A2 (2x1) * B2 (1x9)
    # A2 is row 2 of A (col index 2).
    # B2 is row 2 of B (row index 2).
    # c_i,k = a_i2 b_2,k
    # 18 multiplications
    for i in range(2):
        for k in range(9):
            U[i*3+2][col] = 1
            V[2*9+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
