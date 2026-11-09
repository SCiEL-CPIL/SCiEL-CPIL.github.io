"""
Rank-44 exact tensor decomposition for 2x3x8 matrix multiplication.
Strassen (rank 7) for each 2x2x2 block + outer product (rank 4) for each 2x1x2 block.
4 blocks of (7+4) = 44 total rank.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 8]
_START_TIME = time.time()
TIME_LIMIT = 145.0

def _verify_decomposition(U, V, W, n=2, m=3, p=8):
    """Fast tensor-based verification."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_approx = np.zeros_like(T)
    for r in range(rank):
        T_approx += U_np[:, r, np.newaxis, np.newaxis] * \
                    V_np[:, r][np.newaxis, :, np.newaxis] * \
                    W_np[:, r][np.newaxis, np.newaxis, :]
    
    return np.allclose(T_approx, T, atol=1e-9)

def construct():
    """
    Constructs rank 44 decomposition for 2x3x8 matrix multiplication.
    """
    if time.time() - _START_TIME > TIME_LIMIT:
        # Fallback: return a trivial (but valid) decomposition with rank 48
        n, m, p = 2, 3, 8
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Split p=8 into 4 pairs. For each pair:
    #   Strassen 2x2x2 for j=0,1 (rank 7)
    #   Outer product 2x1x2 for j=2 (rank 4)
    # Total: 4 * (7 + 4) = 44
    
    for k_block in range(4):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        # A indices for j=0,1: a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=3, d=A[1,1]=4
        a_idx, b_idx, c_idx, d_idx = 0, 1, 3, 4
        # B indices for j=0,1: e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
        e_idx, f_idx = k0, k1
        g_idx, h_idx = p + k0, p + k1
        # W indices: w[i,k] = k*n + i
        w00, w01 = k0 * n, k1 * n
        w10, w11 = k0 * n + 1, k1 * n + 1
        
        # M1 = (a+d)(e+h) -> C[0,0]+, C[1,1]+
        U[a_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1; V[h_idx][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # M2 = (c+d)e -> C[1,0]+, C[1,1]-
        U[c_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C[0,1]+, C[1,1]+
        U[a_idx][col] += 1
        V[f_idx][col] += 1; V[h_idx][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # M4 = d(g-e) -> C[0,0]+, C[1,0]+
        U[d_idx][col] += 1
        V[g_idx][col] += 1; V[e_idx][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # M5 = (a+b)h -> C[0,0]-, C[0,1]+
        U[a_idx][col] += 1; U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C[1,1]+
        U[c_idx][col] += 1; U[a_idx][col] -= 1
        V[e_idx][col] += 1; V[f_idx][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C[0,0]+
        U[b_idx][col] += 1; U[d_idx][col] -= 1
        V[g_idx][col] += 1; V[h_idx][col] += 1
        W[w00][col] += 1
        col += 1
        
        # Part 2: Outer product for j=2
        a2_idx, c2_idx = 2, 5
        e2_idx, f2_idx = 2 * p + k0, 2 * p + k1
        
        # C[0,k0] = A[0,2]*B[2,k0]
        U[a2_idx][col] += 1; V[e2_idx][col] += 1; W[w00][col] += 1
        col += 1
        # C[0,k1] = A[0,2]*B[2,k1]
        U[a2_idx][col] += 1; V[f2_idx][col] += 1; W[w01][col] += 1
        col += 1
        # C[1,k0] = A[1,2]*B[2,k0]
        U[c2_idx][col] += 1; V[e2_idx][col] += 1; W[w10][col] += 1
        col += 1
        # C[1,k1] = A[1,2]*B[2,k1]
        U[c2_idx][col] += 1; V[f2_idx][col] += 1; W[w11][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
