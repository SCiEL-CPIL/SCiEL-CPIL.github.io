"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _construct_rank_36():
    """
    Construct rank-36 decomposition for 2x4 x 4x5 matrix multiplication.
    Splits into two 2x2x5 blocks.
    Each 2x2x5 uses Strassen (rank 7) for 2 cols + Schoolbook (rank 4) for 1 col.
    Wait, 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1? No.
    2x2x5 has 5 columns in B.
    Strassen handles 2x2x2 (2 cols of B). Rank 7.
    Remaining 3 cols of B.
    Schoolbook for 2x2x1 is rank 4.
    So 3 cols = 3 * rank 4 = 12.
    Total 7 + 12 = 19.
    Two blocks = 38.
    
    Why did the previous program say 36?
    "Each 2x2x5 splits into two 2x2x2 (Strassen, rank 7) + one 2x2x1 (schoolbook, rank 4). Total rank: 2 * (7 + 7 + 4) = 36."
    This implies 2x2x5 is split into 2x2x2 + 2x2x2 + 2x2x1.
    Sum of cols: 2 + 2 + 1 = 5.
    Rank: 7 + 7 + 4 = 18.
    Two blocks: 36.
    
    This is correct.
    """
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1
        col = col_offset + 0
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        
        # M2
        col = col_offset + 1
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1; W[c22][col] = -1
        
        # M3
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        
        # M4
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1
        
        # M5
        col = col_offset + 4
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1
        
        # M6
        col = col_offset + 5
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1
        
        # M7
        col = col_offset + 6
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1

    def add_schoolbook_221(i1, i2, j1, j2, k1, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b21 = j2 * p + k1
        
        c11 = k1 * n + i1
        c21 = k1 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; V[b11][col] = 1; W[c11][col] = 1
        
        col = col_offset + 1
        U[a12][col] = 1; V[b21][col] = 1; W[c11][col] = 1
        
        col = col_offset + 2
        U[a21][col] = 1; V[b11][col] = 1; W[c21][col] = 1
        
        col = col_offset + 3
        U[a22][col] = 1; V[b21][col] = 1; W[c21][col] = 1

    col = 0
    # Block 1: j=0,1
    add_strassen_222(0, 1, 0, 1, 0, 1, col); col += 7
    add_strassen_222(0, 1, 0, 1, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 0, 1, 4, col); col += 4
    
    # Block 2: j=2,3
    add_strassen_222(0, 1, 2, 3, 0, 1, col); col += 7
    add_strassen_222(0, 1, 2, 3, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 2, 3, 4, col); col += 4
    
    return U, V, W

def construct():
    U, V, W = _construct_rank_36()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
