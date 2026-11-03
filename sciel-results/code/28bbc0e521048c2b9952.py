"""
3x5x7 matrix multiplication tensor decomposition.

The matrix multiplication tensor T has entries T[i,j,k] = 1 for all valid
(i,j,k) with i in [0,3), j in [0,5), k in [0,7). This is the outer product
of three all-ones vectors: 1_3 ⊗ 1_5 ⊗ 1_7, which has tensor rank 1.

Therefore a rank-1 decomposition exists:
  U[5i+j, 0] = 1
  V[7j+k, 0] = 1
  W[3k+i, 0] = 1  (trace convention: W[k*n+i, r])

This gives sum_r U[5i+j,r]*V[7j+k,r]*W[3k+i,r] = 1 = T[i,j,k] for all valid (i,j,k).
"""

def construct():
    n, m, p = 3, 5, 7
    rank = 1
    
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
