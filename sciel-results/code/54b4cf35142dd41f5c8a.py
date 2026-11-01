import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Use two copies of rank-11 decomposition for 2x3x3
    # This gives rank 22 for 2x3x6
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Rank-11 decomposition for 2x3x3 matrix multiplication
    # A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # Row indexing:
    # U: row = i*m + j = i*3 + j (i in 0..1, j in 0..2)
    # V: row = j*p + k = j*3 + k (j in 0..2, k in 0..2)
    # W: row = k*n + i = k*2 + i (k in 0..2, i in 0..1) [trace convention W[k*n+i,r]]
    
    # The 11 rank-1 terms for 2x3x3:
    # Terms 0-5: schoolbook terms for C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2]
    #   But we need to be careful - these 6 terms alone don't cover all 18 entries
    #   We need 5 more terms to complete the decomposition
    
    # Actually, let me use a known valid rank-11 decomposition.
    # The key insight: for 2x3x3, we can use the following structure:
    # - 6 terms for the "diagonal" products
    # - 5 terms that are linear combinations
    
    # Let me use a verified construction:
    # Based on the known optimal rank-11 for 2x3x3
    
    # Factor 0: A[0,0]*B[0,0] -> C[0,0]
    U[0, 0] = 1   # A[0,0]
    V[0, 0] = 1   # B[0,0]
    W[0, 0] = 1   # C[0,0] (k=0,i=0)
    
    # Factor 1: A[0,1]*B[1,0] -> C[0,0]
    U[1, 1] = 1   # A[0,1]
    V[3, 1] = 1   # B[1,0]
    W[0, 1] = 1   # C[0,0]
    
    # Factor 2: A[0,2]*B[2,0] -> C[0,0]
    U[2, 2] = 1   # A[0,2]
    V[6, 2] = 1   # B[2,0]
    W[0, 2] = 1   # C[0,0]
    
    # Factor 3: A[0,0]*B[0,1] -> C[0,1]
    U[0, 3] = 1   # A[0,0]
    V[1, 3] = 1   # B[0,1]
    W[2, 3] = 1   # C[0,1] (k=1,i=0)
    
    # Factor 4: A[0,1]*B[1,1] -> C[0,1]
    U[1, 4] = 1   # A[0,1]
    V[4, 4] = 1   # B[1,1]
    W[2, 4] = 1   # C[0,1]
    
    # Factor 5: A[0,2]*B[2,1] -> C[0,1]
    U[2, 5] = 1   # A[0,2]
    V[7, 5] = 1   # B[2,1]
    W[2, 5] = 1   # C[0,1]
    
    # Factor 6: A[0,0]*B[0,2] -> C[0,2]
    U[0, 6] = 1   # A[0,0]
    V[2, 6] = 1   # B[0,2]
    W[4, 6] = 1   # C[0,2] (k=2,i=0)
    
    # Factor 7: A[0,1]*B[1,2] -> C[0,2]
    U[1, 7] = 1   # A[0,1]
    V[5, 7] = 1   # B[1,2]
    W[4, 7] = 1   # C[0,2]
    
    # Factor 8: A[0,2]*B[2,2] -> C[0,2]
    U[2, 8] = 1   # A[0,2]
    V[8, 8] = 1   # B[2,2]
    W[4, 8] = 1   # C[0,2]
    
    # Factor 9: A[1,0]*B[0,0] -> C[1,0]
    U[3, 9] = 1   # A[1,0]
    V[0, 9] = 1   # B[0,0]
    W[1, 9] = 1   # C[1,0] (k=0,i=1)
    
    # Factor 10: A[1,1]*B[1,0] -> C[1,0]
    U[4, 10] = 1  # A[1,1]
    V[3, 10] = 1  # B[1,0]
    W[1, 10] = 1  # C[1,0]
    
    # Factor 11: A[1,2]*B[2,0] -> C[1,0]
    U[5, 11] = 1  # A[1,2]
    V[6, 11] = 1  # B[2,0]
    W[1, 11] = 1  # C[1,0]
    
    # Factor 12: A[1,0]*B[0,1] -> C[1,1]
    U[3, 12] = 1  # A[1,0]
    V[1, 12] = 1  # B[0,1]
    W[3, 12] = 1  # C[1,1] (k=1,i=1)
    
    # Factor 13: A[1,1]*B[1,1] -> C[1,1]
    U[4, 13] = 1  # A[1,1]
    V[4, 13] = 1  # B[1,1]
    W[3, 13] = 1  # C[1,1]
    
    # Factor 14: A[1,2]*B[2,1] -> C[1,1]
    U[5, 14] = 1  # A[1,2]
    V[7, 14] = 1  # B[2,1]
    W[3, 14] = 1  # C[1,1]
    
    # Factor 15: A[1,0]*B[0,2] -> C[1,2]
    U[3, 15] = 1  # A[1,0]
    V[2, 15] = 1  # B[0,2]
    W[5, 15] = 1  # C[1,2] (k=2,i=1)
    
    # Factor 16: A[1,1]*B[1,2] -> C[1,2]
    U[4, 16] = 1  # A[1,1]
    V[5, 16] = 1  # B[1,2]
    W[5, 16] = 1  # C[1,2]
    
    # Factor 17: A[1,2]*B[2,2] -> C[1,2]
    U[5, 17] = 1  # A[1,2]
    V[8, 17] = 1  # B[2,2]
    W[5, 17] = 1  # C[1,2]
    
    # This is schoolbook rank 18 for 2x3x3, not rank 11
    # Let me reset and use a proper approach
    
    # For 2x3x6, let's use a block decomposition:
    # B is 3x6, split into B1 (3x3, columns 0-2) and B2 (3x3, columns 3-5)
    # C = A * [B1 B2] = [A*B1, A*B2]
    # 
    # For A*B1 (2x3 * 3x3 -> 2x3), we need rank-11
    # For A*B2 (2x3 * 3x3 -> 2x3), we need rank-11
    # Total rank = 22
    
    # Let me construct a proper rank-11 decomposition for 2x3x3
    # Using the known optimal decomposition
    
    # Reset
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Rank-11 decomposition for 2x3x3 (verified construction)
    # The 11 factors are:
    # 6 "diagonal" terms + 5 "off-diagonal" combination terms
    
    # For the first block (columns 0-2 of B, columns 0-2 of C):
    # Factor 0: A[0,0]*B[0,0] -> C[0,0]
    U[0, 0] = 1
    V[0, 0] = 1    # B[0,0] (row 0*3+0)
    W[0, 0] = 1    # C[0,0] (row 0*2+0)
    
    # Factor 1: A[0,1]*B[1,0] -> C[0,0]
    U[1, 1] = 1
    V[3, 1] = 1    # B[1,0] (row 1*3+0)
    W[0, 1] = 1
    
    # Factor 2: A[0,2]*B[2,0] -> C[0,0]
    U[2, 2] = 1
    V[6, 2] = 1    # B[2,0] (row 2*3+0)
    W[0, 2] = 1
    
    # Factor 3: A[0,0]*B[0,1] -> C[0,1]
    U[0, 3] = 1
    V[1, 3] = 1    # B[0,1] (row 0*3+1)
    W[2, 3] = 1    # C[0,1] (row 1*2+0)
    
    # Factor 4: A[0,1]*B[1,1] -> C[0,1]
    U[1, 4] = 1
    V[4, 4] = 1    # B[1,1] (row 1*3+1)
    W[2, 4] = 1
    
    # Factor 5: A[0,2]*B[2,1] -> C[0,1]
    U[2, 5] = 1
    V[7, 5] = 1    # B[2,1] (row 2*3+1)
    W[2, 5] = 1
    
    # Factor 6: A[0,0]*B[0,2] -> C[0,2]
    U[0, 6] = 1
    V[2, 6] = 1    # B[0,2] (row 0*3+2)
    W[4, 6] = 1    # C[0,2] (row 2*2+0)
    
    # Factor 7: A[0,1]*B[1,2] -> C[0,2]
    U[1, 7] = 1
    V[5, 7] = 1    # B[1,2] (row 1*3+2)
    W[4, 7] = 1
    
    # Factor 8: A[0,2]*B[2,2] -> C[0,2]
    U[2, 8] = 1
    V[8, 8] = 1    # B[2,2] (row 2*3+2)
    W[4, 8] = 1
    
    # Factor 9: A[1,0]*B[0,0] -> C[1,0]
    U[3, 9] = 1
    V[0, 9] = 1
    W[1, 9] = 1    # C[1,0] (row 0*2+1)
    
    # Factor 10: A[1,1]*B[1,0] -> C[1,0]
    U[4, 10] = 1
    V[3, 10] = 1
    W[1, 10] = 1
    
    # This is only 11 factors but it's still schoolbook-like
    # I need a proper rank-11 with cancellations
    
    # Let me use a different, verified approach
    # For 2x3x3, the optimal rank-11 can be constructed as follows:
    # Use the fact that we can combine terms
    
    # Actually, let me just use the schoolbook for 2x3x6 with rank 36
    # and return that as a safe baseline
    rank = n * m * p  # 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
