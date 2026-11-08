import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = 2, 3, 4
    # Tensor T[i,j,k] = 1 where i=row of A (0..n-1), j=col of B (0..p-1), k=inner (0..m-1)
    # W index is k*n + i, so for fixed k,i, W has entries at positions k*n+i
    # Actually W[k*n+i, r] contributes to C[i, j] via V[k*p+j, r]
    
    # Let's define the tensor properly:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # In decomposition: C[i,j] = sum_r U[i*m+k... wait, U has n*m rows
    # U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r]... no
    # W[k*n + i, r] - so W is indexed by (k, i) where k is inner, i is row
    # So C[i,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[k*n+i, r]
    
    # For the standard decomposition, each term (i,j,k) has:
    # U[i*m+k, r] = 1, V[k*p+j, r] = 1, W[k*n+i, r] = 1
    # But wait - W[k*n+i, r] appears for ALL j in the same term? No, each (i,j,k) is a separate rank-1 term.
    
    # Actually, for the standard decomposition:
    # For each (i, j, k): create a rank-1 term where
    # U[i*m+k] = 1, V[k*p+j] = 1, W[k*n+i] = 1 (in that rank column)
    # This gives rank = n*m*p = 24
    
    # But we can do better. Let's try to find a lower rank decomposition.
    # The bilinear complexity of 2x3x4 matrix multiplication...
    
    # Let's just do the standard rank-24 decomposition first, which is guaranteed correct.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                U[i * m + k][r] = 1
                V[k * p + j][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
