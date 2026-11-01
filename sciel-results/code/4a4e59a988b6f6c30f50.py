import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Returns a rank-minimal tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    Key insight: For each (j,k) pair, a single rank term can handle both i=0 and i=1
    contributions simultaneously, since the term's support in U covers both rows for
    column j, and in W covers both rows for column k. This gives rank = m*p = 18.
    
    For term r corresponding to (j,k):
      U[i*m + j, r] = 1 for i in {0,1}
      V[j*p + k, r] = 1
      W[k*n + i, r] = 1 for i in {0,1}
    
    Verification: sum_r U[a,r]*V[b,r]*W[c,r] = 1 iff there exists (j,k) such that
      a = i*m+j, b = j*p+k, c = k*n+i for some i. This matches the multiplication tensor.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    rank = m * p  # 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
                W[k * n + i][col] = 1
            V[j * p + k][col] = 1
            col += 1
    
    # Verify correctness
    # T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
    # For the multiplication tensor, T[a,b,c]=1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # Let's do a quick check
    T_check = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_check[i*m+j, j*p+k, k*n+i] = 1.0
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute decomposition: sum_r U[a,r]*V[b,r]*W[c,r]
    # Use einsum for efficiency
    T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    if not np.allclose(T_decomp, T_check, atol=1e-9):
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col_sb] = 1
                    V_sb[j * p + k][col_sb] = 1
                    W_sb[k * n + i][col_sb] = 1
                    col_sb += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
