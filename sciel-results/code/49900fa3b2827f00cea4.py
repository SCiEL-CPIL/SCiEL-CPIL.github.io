"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    # Start with schoolbook rank 36
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
    
    # Try to reduce rank by eliminating dependent terms
    # For 3x3x4, try rank 24 using a structured decomposition
    # The idea: group by middle index j, and for each j, we have a 3x4 outer product
    # We can potentially combine terms across j if there's structure
    
    # Try a rank-24 decomposition: use the fact that for each pair (i,k), 
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We can try to find a decomposition of rank < 36
    
    # Known result: rank(3x3x4) <= 24 using a specific construction
    # Let's try to construct a rank-24 decomposition
    
    rank_target = 24
    U2 = [[0] * rank_target for _ in range(n * m)]
    V2 = [[0] * rank_target for _ in range(m * p)]
    W2 = [[0] * rank_target for _ in range(p * n)]
    
    # Structure: for each j in {0,1,2}, we have 3x4 = 12 terms, total 36
    # To get to 24, we need to eliminate 12 terms
    # Try: for j=0 and j=1, use all 12 terms each (24 total), and handle j=2 differently
    # Actually, let's try a different approach: use a basis transformation
    
    # Alternative: use the "ladder" decomposition
    # For 3x3x4, we can try rank = n*p + (m-1)*n*p = 3*4 + 2*3*4 = 12+24=36, no improvement
    
    # Let's just return the schoolbook for now and verify it's correct
    # The key is to find a valid lower rank decomposition
    
    # Try rank 20: use the structure that C[i,k] depends on all j
    # A rank-r decomposition has r terms, each is a tensor product u* v* w*
    # where u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    
    # For now, return schoolbook rank 36
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
