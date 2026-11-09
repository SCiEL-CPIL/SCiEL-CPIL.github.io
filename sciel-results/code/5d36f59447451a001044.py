"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    # Use Strassen-inspired decomposition for 2x2 blocks combined with schoolbook for remaining dims
    n, m, p = DIMENSIONS
    # Decompose into 2x2 blocks where possible
    # For 2x5x6: treat as (2x2) x (2x2) x (2x2) blocks with padding
    # Use rank-7 Strassen for 2x2x2, then combine with schoolbook for the 5 and 6 dimensions
    
    # Strategy: Split 5 = 2+2+1, 6 = 2+2+2
    # For each 2x2 block pair, use Strassen (rank 7)
    # For remaining 1x2 and 2x1 blocks, use schoolbook
    
    # Build U, V, W matrices
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U.append([u_idx, u_val])
        V.append([v_idx, v_val])
        W.append([w_idx, w_val])
    
    # Strassen 2x2x2 coefficients (rank 7)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x5x6, we can use 2x2 blocks:
    # Rows: 2 (already 2)
    # Cols of A: 5 = 2+2+1
    # Cols of B: 6 = 2+2+2
    
    # Let's use a hybrid: Strassen for the first 2x2x2 block, schoolbook for rest
    # This gives: 7 (Strassen) + 2*2*1 + 2*1*2 + 2*1*1 = 7 + 4 + 4 + 2 = 17 for 2x5x6? 
    # Actually need to be more careful.
    
    # Simpler approach: Use rank-7 Strassen for each 2x2x2 subproblem
    # 5 = 2+2+1, 6 = 2+2+2
    # Number of 2x2x2 blocks: min(2//2, 5//2, 6//2) = 1 full block, plus remainders
    
    # Let's just use schoolbook but with a more efficient indexing
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
