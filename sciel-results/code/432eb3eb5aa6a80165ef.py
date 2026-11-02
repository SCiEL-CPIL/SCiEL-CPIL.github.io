import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    def verify(U, V, W, n, m, p):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try a rank-48 decomposition using proper block structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split into 4 blocks: A is 2x2 blocks, B is 2x3 blocks
    # Block (ri, cj): A[ri*2:(ri+1)*2, cj*2:(cj+1)*2] * B[ri*2:(ri+1)*2, :] 
    # Wait, that's not right. Let me think again.
    #
    # Actually for matrix mult C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    #
    # If we split the j index into two halves j in {0,1} and j in {2,3}:
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #
    # The first term uses A[:, 0:2] (4x2) and B[0:2, :] (2x6)
    # The second term uses A[:, 2:4] (4x2) and B[2:4, :] (2x6)
    #
    # Each term is a 4x2 * 2x6 multiplication.
    # Schoolbook rank for 4x2 * 2x6: 4*2*6 = 48
    # Total: 48 + 48 = 96. Same as before.
    #
    # To do better, we need to share between the two halves.
    # Let's try: for each i, k, C[i,k] = sum_j A[i,j]*B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # We can use the identity: (x+y)(u+v) = xu + yv + (x+y)(u+v) - (xu+yv)
    # But that doesn't help directly.
    #
    # Let me just go with schoolbook rank 96 and verify it works.
    
    rank = n * m * p  # 96
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    
    if not verify(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook verification failed")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
