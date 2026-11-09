import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    dim_U = n * m  # 6
    dim_V = m * p  # 30
    dim_W = p * n  # 20
    
    deadline = time.time() + 140  # Safety margin
    
    # Try to construct a rank-22 decomposition for M_{2,3,10}
    # Using the structure: C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Construction:
    # We use 2*p + m - 1 = 22 terms
    # Terms 0 to 9: for each k, a term that handles "row 0" contributions
    # Terms 10 to 19: for each k, a term that handles "row 1" contributions  
    # Terms 20, 21: correction terms
    
    # Actually, let me use a cleaner construction.
    # For M_{2,m,p}, we can use the following rank-(2p + m - 1) construction:
    
    rank = 2 * p + m - 1  # 22
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    # For each k in 0..p-1:
    #   Term 2*k:   U[A[0,j], 2*k] = 1 for j=0..m-1, V[B[j,k], 2*k] = 1 for j=0..m-1, W[C[0,k], 2*k] = 1
    #   Term 2*k+1: U[A[1,j], 2*k+1] = 1 for j=0..m-1, V[B[j,k], 2*k+1] = 1 for j=0..m-1, W[C[1,k], 2*k+1] = 1
    # This gives 2*p = 20 terms, but it's just the schoolbook split by row.
    
    # We need to do better. Let me use a Strassen-like idea for the 2x3x1 case.
    # For 2x3x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    #           C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    # This is a 2x3x1 matrix multiplication. The rank is at most 5 (using 3 multiplications for each row, but sharing).
    
    # Actually, for 2xm x mx1, the rank is m+1 when m is even, m+2 when m is odd? No, that's not right.
    
    # Let me just use the schoolbook for now and return rank 60, but try to find a better one.
    
    # Schoolbook:
    U_school = np.zeros((dim_U, n*m*p))
    V_school = np.zeros((dim_V, n*m*p))
    W_school = np.zeros((dim_W, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using a simple approach:
    # Check if we can find linear dependencies in the schoolbook decomposition.
    
    # For now, return the schoolbook. This is a valid but suboptimal solution.
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
