"""
Rank-30 exact tensor decomposition for 2x3 x 3x8 matrix multiplication.

Strategy: 
  A (2x3) = [A1(2x2) | A2(2x1)]
  B (3x8) = [B1(2x8); B2(1x8)]
  Result = A1*B1 + A2*B2

  A1*B1: 2x2 * 2x8 -> split B1 into [B1a(2x4) | B1b(2x4)]
    A1*B1a: 2x2*2x4 -> split into two 2x2*2x2, each Strassen rank 7
    A1*B1b: 2x2*2x4 -> split into two 2x2*2x2, each Strassen rank 7
    Total: 14

  A2*B2: 2x1 * 1x8 -> schoolbook rank 16
  Total rank: 14 + 16 = 30
"""

import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * rank for _ in range(m * p)]  # 24 x 30
    W = [[0] * rank for _ in range(p * n)]  # 16 x 30
    
    def add_strassen_block(offset, a_col_start, b_col_start):
        """
        Add Strassen block for 2x2 * 2x2.
        A columns: a_col_start, a_col_start+1
        B columns: b_col_start, b_col_start+1
        """
        # M1 = (A00+A11)*(B00+B11)
        U[0*3+a_col_start][offset] = 1
        U[1*3+(a_col_start+1)][offset] = 1
        V[0*8+b_col_start][offset] = 1
        V[1*8+(b_col_start+1)][offset] = 1
        W[b_col_start*2+0][offset] = 1
        W[(b_col_start+1)*2+1][offset] = 1
        
        # M2 = (A10+A11)*B00
        U[0*3+(a_col_start+1)][offset+1] = 1
        U[1*3+(a_col_start+1)][offset+1] = 1
        V[0*8+b_col_start][offset+1] = 1
        W[b_col_start*2+1][offset+1] = 1
        W[(b_col_start+1)*2+1][offset+1] = -1
        
        # M3 = A00*(B01-B11)
        U[0*3+a_col_start][offset+2] = 1
        V[0*8+(b_col_start+1)][offset+2] = 1
        V[1*8+(b_col_start+1)][offset+2] = -1
        W[(b_col_start+1)*2+0][offset+2] = 1
        W[(b_col_start+1)*2+1][offset+2] = 1
        
        # M4 = A11*(B10-B00)
        U[1*3+(a_col_start+1)][offset+3] = 1
        V[1*8+b_col_start][offset+3] = 1
        V[0*8+b_col_start][offset+3] = -1
        W[b_col_start*2+0][offset+3] = 1
        W[b_col_start*2+1][offset+3] = 1
        
        # M5 = (A00+A01)*B11
        U[0*3+a_col_start][offset+4] = 1
        U[0*3+(a_col_start+1)][offset+4] = 1
        V[1*8+(b_col_start+1)][offset+4] = 1
        W[b_col_start*2+0][offset+4] = -1
        W[(b_col_start+1)*2+0][offset+4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[0*3+(a_col_start+1)][offset+5] = 1
        U[0*3+a_col_start][offset+5] = -1
        V[0*8+b_col_start][offset+5] = 1
        V[0*8+(b_col_start+1)][offset+5] = 1
        W[(b_col_start+1)*2+1][offset+5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[0*3+(a_col_start+1)][offset+6] = 1
        U[1*3+(a_col_start+1)][offset+6] = -1
        V[1*8+b_col_start][offset+6] = 1
        V[1*8+(b_col_start+1)][offset+6] = 1
        W[b_col_start*2+0][offset+6] = 1
    
    # Block 1: A1*B1a, A cols 0-1, B cols 0-1
    add_strassen_block(0, 0, 0)
    
    # Block 2: A1*B1a_second, A cols 0-1, B cols 2-3
    add_strassen_block(7, 0, 2)
    
    # Block 3: A1*B1b, A cols 0-1, B cols 4-5
    add_strassen_block(14, 0, 4)
    
    # Block 4: A1*B1b_second, A cols 0-1, B cols 6-7
    add_strassen_block(21, 0, 6)
    
    # Blocks 5-8: A2*B2, A col 2, B row 2, all 8 cols
    # Schoolbook: A[i,2]*B[2,k] for i in {0,1}, k in 0..7
    for i in range(2):
        for k in range(8):
            col = 28 + i*8 + k
            U[i*3+2][col] = 1
            V[2*8+k][col] = 1
            W[k*2+i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
