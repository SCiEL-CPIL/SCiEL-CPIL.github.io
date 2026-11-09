import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # For 2x3 * 3x9 matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    # We need: sum_t U[a,t] * V[b,t] * W[c,t] = 1 for all (a,b,c) in support
    
    # The rank of matrix multiplication tensor for (n x m) * (m x p) is at most n*m*p
    # but can be lower. For our case, n=2, m=3, p=9, so n*m*p = 54.
    
    # Key insight: We can achieve rank 27 by exploiting the structure.
    # For each (j,k) pair (3*9=27 terms), we can construct rank-1 terms.
    
    # Let's think about this more carefully:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # 
    # For a rank-r decomposition, we need r triples (u_r, v_r, w_r) such that
    # sum_{t=0}^{r-1} u_t[a] * v_t[b] * w_t[c] = T[a,b,c]
    #
    # The support of T consists of (a,b,c) = (i*m+j, j*p+k, k*n+i) for all i,j,k
    # 
    # For rank 27, let's index t by (j', k') where j' in {0,1,2}, k' in {0,...,8}
    
    r = 27
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    # For each (j', k') pair, define:
    # U[i*m+j, t] = 1 if j == j', else 0
    # V[j'*p+k', t] = 1, all others 0
    # W[k'*n+i, t] = 1/n for all i
    
    # Check: sum_t U[a,t]*V[b,t]*W[c,t] where a=i*m+j, b=j'*p+k', c=k*n+i
    # Only t=(j,k') contributes: U[a,t]=1 (if j==j'), V[b,t]=1 (if b=j'*p+k')
    # W[c,t] = 1/n
    # But we need the sum to be 1, and we only get 1/n.
    # 
    # Wait, this doesn't work directly. Let me reconsider.
    
    # Actually, the tensor entry T[i*m+j, j*p+k, k*n+i] = 1
    # For the decomposition to work, for each (a,b,c) in the support, we need exactly
    # one t such that U[a,t]=V[b,t]=W[c,t]=1 (or appropriate values).
    
    # Let's try a different approach. The standard approach for matrix multiplication:
    # For (n x m) * (m x p), the rank is at most n*m*p, achieved by schoolbook.
    # 
    # Can we do better? For 2x3 * 3x9:
    # 
    # Actually, let's just return the schoolbook rank-54 solution which is guaranteed correct.
    # The minimal rank for general matrix multiplication is unknown, but 54 is safe.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
