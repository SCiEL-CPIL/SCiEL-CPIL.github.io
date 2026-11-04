import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # We need to decompose the matrix multiplication tensor for 4x4 * 4x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # where A is 4x4, B is 4x7, C is 4x7
    # 
    # The tensor has entries T[i,j,k] = 1 for all valid (i,j,k)
    # A rank-1 decomposition term is u ⊗ v ⊗ w where:
    #   u is indexed by (i,j) pairs (size n*m = 16)
    #   v is indexed by (j,k) pairs (size m*p = 28)  
    #   w is indexed by (k,i) pairs (size p*n = 28)
    # 
    # We need: sum_c U[c,i,j] * V[c,j,k] * W[c,k,i] = 1 for all i,j,k
    # 
    # Strategy: Use Strassen for the 4x4 * 4x4 part (first 4 columns of B)
    # and schoolbook for the 4x4 * 4x3 part (last 3 columns of B)
    # 
    # Actually, let me think more carefully. The matrix multiplication is:
    # C = A * B where A is 4x4 and B is 4x7
    # 
    # We can split B into B1 (4x4) and B2 (4x3):
    # C = A * B1 + A * B2
    # 
    # For A * B1 (4x4 * 4x4), we can use Strassen's algorithm which has rank 7
    # For A * B2 (4x4 * 4x3), we use schoolbook with rank 4*4*3 = 48
    # 
    # But wait, Strassen gives rank 7 for 2x2 * 2x2. For 4x4 * 4x4, we can
    # apply Strassen at the block level. Let's think about this differently.
    # 
    # Actually, let me use a simpler approach. The schoolbook rank is n*m*p = 4*4*7 = 112.
    # 
    # Let me try to find a better decomposition. One approach is to use the fact that
    # matrix multiplication can be expressed using fewer multiplications.
    # 
    # For now, let me implement the schoolbook decomposition and verify it works.
    # Then I'll try to optimize.
    
    terms = []
    
    def make_u(idx_i, idx_j, val):
        u = [0] * (n * m)
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val):
        v = [0] * (m * p)
        v[idx_j * p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val):
        w = [0] * (p * n)
        w[idx_k * n + idx_i] = val
        return w
    
    # Schoolbook: rank 112
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = make_u(i, j, 1)
                v = make_v(j, k, 1)
                w = make_w(k, i, 1)
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
