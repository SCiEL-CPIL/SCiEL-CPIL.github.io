"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 78
    rank = 78
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (1x4) * B (4x7) -> C1 (1x7)
    # Rank 28
    for j in range(4):
        for k in range(7):
            U[0 * 4 + j][col] = 1
            V[j * 7 + k][col] = 1
            W[k * 3 + 0][col] = 1
            col += 1
            
    # Part 2: A2 (2x4) * B (4x7) -> C2 (2x7)
    # Split B into BL (2x7) and BR (2x7)
    # A2 = [A21, A22] where A21, A22 are 2x2
    # C2 = A21 * BL + A22 * BR
    # Each 2x2 * 2x7 has rank 25 (using 7+7+4)
    # Total rank 50
    
    # Helper to add 2x2 * 2xp decomposition
    def add_2x2_2xp(U, V, W, col, i_offset, j_offset, k_offset, p_size):
        # i_offset: row offset in C (0 or 1 for the 2 rows)
        # j_offset: col offset in A (0 or 1 for the 2 cols)
        # k_offset: col offset in B (0 or 1 for the 2 cols)
        # p_size: number of columns in B (7)
        
        # We use 7 terms for first 2 cols, 7 for next 2, 4 for last 3?
        # No, 7+7+4 = 18 for 7 cols?
        # R(2,2,2)=7, R(2,2,2)=7, R(2,2,3)=?
        # Let's use 7+7+7+4 = 25 for 2+2+2+1
        # Cols: 0,1 (7 terms), 2,3 (7 terms), 4,5 (7 terms), 6 (4 terms)
        
        # Term 1: Cols 0,1 (2x2 * 2x2)
        # Strassen decomposition for 2x2 * 2x2
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # This is for specific indices.
        # We need to map this to the global indices.
        
        # This is too complex to hardcode correctly without verification.
        # I will fall back to schoolbook for the sub-blocks if I can't verify.
        # Schoolbook for 2x2 * 2x7 is 4*7 = 28.
        # 28 + 28 = 56.
        # Total 28 + 56 = 84.
        # This is not an improvement.
        
        # I will return the schoolbook decomposition to ensure validity.
        pass

    # Since I cannot guarantee a valid lower rank decomposition quickly, 
    # I will return the schoolbook decomposition.
    # However, the score 1.0 is for 84.
    # I must try to beat it.
    
    # Let's try a simple heuristic: 
    # Use 0s and 1s randomly and check? No, too slow.
    
    # I will return the schoolbook decomposition.
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
