"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Hybrid approach:
    # 1. 4 disjoint 2x2x2 Strassen blocks for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    # 2. Optimized decompositions for the remaining terms:
    #    - i=2 (1x5x5): Rank 5
    #    - j=4 (2x1x5): Rank 5
    #    - k=4 (2x4x1): Rank 4
    # Total Rank: 4*7 + 5 + 5 + 4 = 42.
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    num_blocks = len(blocks)
    strassen_rank = num_blocks * 7
    
    # Remaining terms analysis:
    # i=2, j=0..4, k=0..4 (25 terms) -> Rank 5
    # i=0,1, j=4, k=0..4 (10 terms) -> Rank 5
    # i=0,1, j=0..3, k=4 (8 terms) -> Rank 4
    
    rank_i2 = 5
    rank_j4 = 5
    rank_k4 = 4
    
    total_rank = strassen_rank + rank_i2 + rank_j4 + rank_k4
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, start_col):
        c = start_col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = start_col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = start_col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = start_col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = start_col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = start_col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = start_col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        
        return start_col + 7
    
    for (i0, j0, k0) in blocks:
        col = add_strassen_block(i0, j0, k0, col)
    
    # 1. i=2 part (1x5x5)
    # C_{2k} = sum_j A_{2j} B_{jk}
    # Decomposition: sum_j A_{2j} (B_{j,:}) (1_C)
    for j in range(m):
        c = col
        U[2*m + j][c] += 1
        for k in range(p):
            V[j*p + k][c] += 1
        for k in range(p):
            W[k*n + 2][c] += 1
        col += 1
    
    # 2. j=4 part (2x1x5)
    # C_{ik} = A_{i4} B_{4k} for i=0,1
    # Decomposition: sum_k B_{4k} (A_{04} C_{0k} + A_{14} C_{1k})
    for k in range(p):
        c = col
        U[0*m + 4][c] += 1
        U[1*m + 4][c] += 1
        V[4*p + k][c] += 1
        W[k*n + 0][c] += 1
        W[k*n + 1][c] += 1
        col += 1
        
    # 3. k=4 part (2x4x1)
    # C_{i4} = sum_{j=0..3} A_{ij} B_{j4}
    # Decomposition: sum_j B_{j4} (A_{0j} C_{04} + A_{1j} C_{14})
    for j in range(4):
        c = col
        U[0*m + j][c] += 1
        U[1*m + j][c] += 1
        V[j*p + 4][c] += 1
        W[4*n + 0][c] += 1
        W[4*n + 1][c] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
