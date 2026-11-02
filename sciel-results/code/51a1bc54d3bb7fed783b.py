import numpy as np
import time

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # 4 blocks of 2 columns (using Strassen, rank 7)
    # 1 block of 1 column (using schoolbook, rank 4)
    # Total rank = 4 * 7 + 4 = 32
    
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for 2 columns
    def add_strassen(col_start, rank_offset):
        # A is 2x2, B is 2x2 (cols col_start, col_start+1)
        # C is 2x2
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B11
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B00
        # M6 = (A10+A01)(B11-B01)
        # M7 = (A00-A10)(B10+B11)
        
        # U indices: 0:A00, 1:A01, 2:A10, 3:A11
        # V indices: 0:B00, 1:B01, 2:B10, 3:B11 (local) -> Global: col_start, col_start+1
        # W indices: 0:C00, 1:C01, 2:C10, 3:C11 (local) -> Global: col_start, col_start+1
        
        # M1
        U[0][rank_offset + col] = 1
        U[3][rank_offset + col] = 1
        V[col_start][rank_offset + col] = 1
        V[col_start+1][rank_offset + col] = 1
        W[col_start][rank_offset + col] = 1
        W[col_start+1][rank_offset + col] = 1
        
        # M2
        U[2][rank_offset + col] = 1
        U[3][rank_offset + col] = 1
        V[col_start+1][rank_offset + col] = 1
        W[col_start+1][rank_offset + col] = 1
        
        # M3
        U[0][rank_offset + col] = 1
        V[col_start+1][rank_offset + col] = 1
        V[col_start][rank_offset + col] = -1
        W[col_start+1][rank_offset + col] = 1
        
        # M4
        U[3][rank_offset + col] = 1
        V[col_start][rank_offset + col] = 1
        V[col_start+1][rank_offset + col] = -1
        W[col_start][rank_offset + col] = 1
        
        # M5
        U[0][rank_offset + col] = 1
        U[1][rank_offset + col] = 1
        V[col_start][rank_offset + col] = 1
        W[col_start][rank_offset + col] = -1
        W[col_start+1][rank_offset + col] = 1
        
        # M6
        U[1][rank_offset + col] = 1
        U[2][rank_offset + col] = 1
        V[col_start+1][rank_offset + col] = 1
        V[col_start][rank_offset + col] = -1
        W[col_start+1][rank_offset + col] = 1
        
        # M7
        U[0][rank_offset + col] = 1
        U[2][rank_offset + col] = -1
        V[col_start][rank_offset + col] = 1
        V[col_start+1][rank_offset + col] = 1
        W[col_start][rank_offset + col] = 1
        W[col_start+1][rank_offset + col] = 1
        
    # Helper to add schoolbook for 1 column
    def add_schoolbook_1col(col_idx, rank_offset):
        # A 2x2, B 2x1, C 2x1
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        # 4 terms
        
        # T1: A00 B00
        U[0][rank_offset + col] = 1
        V[col_idx][rank_offset + col] = 1
        W[col_idx][rank_offset + col] = 1
        
        # T2: A01 B10
        U[1][rank_offset + col] = 1
        V[col_idx+1][rank_offset + col] = 1
        W[col_idx][rank_offset + col] = 1
        
        # T3: A10 B00
        U[2][rank_offset + col] = 1
        V[col_idx][rank_offset + col] = 1
        W[col_idx+1][rank_offset + col] = 1
        
        # T4: A11 B10
        U[3][rank_offset + col] = 1
        V[col_idx+1][rank_offset + col] = 1
        W[col_idx+1][rank_offset + col] = 1

    # Apply Strassen to cols 0,1; 2,3; 4,5; 6,7
    for i in range(4):
        add_strassen(2*i, i*7)
        
    # Apply schoolbook to col 8
    add_schoolbook_1col(8, 28)
    
    return {"U": U, "V": V, "W": W}
