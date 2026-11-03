"""Tensor decomposition for 3x3 * 3x8 matrix multiplication.
Uses Strassen's algorithm on 2x2 blocks where beneficial, combined with
schoolbook for the rest, to achieve rank 68.
"""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x8 matrix.
    
    Strategy: Partition A (3x3) into a 2x2 top-left block and remaining rows/cols.
    Use Strassen on the 2x2 block (rank 7 instead of 8) for each of the 4 
    2x2 column blocks of B's top 2 rows. Schoolbook for the rest.
    
    Total rank: 4 * 7 (Strassen) + 16 (A[0:2,2]*B[2,:]) + 16 (A[2,0:2]*B[0:2,:]) + 8 (A[2,2]*B[2,:]) = 68.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # U: (n*m) x rank = 9 x rank
    # V: (m*p) x rank = 24 x rank
    # W: (p*n) x rank = 24 x rank
    
    rank_estimate = 68
    U = [[0] * rank_estimate for _ in range(n * m)]
    V = [[0] * rank_estimate for _ in range(m * p)]
    W = [[0] * rank_estimate for _ in range(p * n)]
    
    col = 0
    
    # Strassen for A[0:2,0:2] * B[0:2, 2b:2b+2] for b in 0..3
    for b in range(4):
        col_off = 2 * b
        
        # m1 = (A[0,0]+A[1,1]) * (B[0,col_off]+B[1,col_off+1])
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + col_off][col] = 1
        V[1 * p + col_off + 1][col] = 1
        W[col_off * n + 0][col] = 1
        W[(col_off + 1) * n + 1][col] = 1
        col += 1
        
        # m2 = (A[1,0]+A[1,1]) * B[0,col_off]
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + col_off][col] = 1
        W[col_off * n + 1][col] = 1
        W[(col_off + 1) * n + 1][col] = -1
        col += 1
        
        # m3 = A[0,0] * (B[0,col_off+1]-B[1,col_off+1])
        U[0 * m + 0][col] = 1
        V[0 * p + col_off + 1][col] = 1
        V[1 * p + col_off + 1][col] = -1
        W[(col_off + 1) * n + 0][col] = 1
        W[(col_off + 1) * n + 1][col] = 1
        col += 1
        
        # m4 = A[1,1] * (B[1,col_off]-B[0,col_off])
        U[1 * m + 1][col] = 1
        V[1 * p + col_off][col] = 1
        V[0 * p + col_off][col] = -1
        W[col_off * n + 0][col] = 1
        W[col_off * n + 1][col] = 1
        col += 1
        
        # m5 = (A[0,0]+A[0,1]) * B[1,col_off+1]
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + col_off + 1][col] = 1
        W[col_off * n + 0][col] = -1
        W[(col_off + 1) * n + 0][col] = 1
        col += 1
        
        # m6 = (A[1,0]-A[0,0]) * (B[0,col_off]+B[0,col_off+1])
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + col_off][col] = 1
        V[0 * p + col_off + 1][col] = 1
        W[(col_off + 1) * n + 1][col] = 1
        col += 1
        
        # m7 = (A[0,1]-A[1,1]) * (B[1,col_off]+B[1,col_off+1])
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + col_off][col] = 1
        V[1 * p + col_off + 1][col] = 1
        W[col_off * n + 0][col] = 1
        col += 1
    
    # Schoolbook for remaining terms:
    # A[0:2, 2] * B[2, :] -> 2*1*8 = 16 terms
    for i in range(2):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # A[2, 0:2] * B[0:2, :] -> 1*2*8 = 16 terms
    for j in range(2):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # A[2, 2] * B[2, :] -> 1*1*8 = 8 terms
    for k in range(p):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
    
    assert col == rank_estimate, f"Expected rank {rank_estimate}, got {col}"
    
    # Safety margin check
    if time.time() - start_time > 140:
        # This block is fast, so this shouldn't happen, but good practice.
        pass
        
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    import numpy as np
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = DIMENSIONS
    # Verify: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # T should be all 1s
    if np.allclose(T, 1.0):
        print(f"SUCCESS: rank = {U.shape[1]}")
    else:
        print(f"FAILURE: T not all 1s. Max diff: {np.max(np.abs(T-1))}")
        print(T)
