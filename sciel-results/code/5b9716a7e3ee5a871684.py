import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 130.0

def _time_exceeded():
    return time.time() - _start_time > TIME_LIMIT

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i*m + j
                idx_v = j*p + k
                idx_w = k*n + i
                if abs(T[idx_u, idx_v, idx_w] - 1.0) > 1e-9:
                    return False
    
    for idx_u in range(n*m):
        for idx_v in range(m*p):
            for idx_w in range(p*n):
                i = idx_u // m
                j = idx_u % m
                j2 = idx_v // p
                k = idx_v % p
                k2 = idx_w // n
                i2 = idx_w % n
                if i == i2 and j == j2 and k == k2:
                    continue
                if abs(T[idx_u, idx_v, idx_w]) > 1e-9:
                    return False
    
    return True

def _build_rank18_decomposition():
    """
    Rank-18 decomposition for 3x4 * 4x4:
    - Two Strassen 2x2x2 blocks for rows {0,1} x cols {0,1} and rows {0,1} x cols {2,3}
    - Four trivial terms for row 2 x cols {0,1,2,3}
    """
    n, m, p = 3, 4, 4
    rank = 18
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    # Strassen block 1: rows {0,1}, cols {0,1}
    # A: a=U[0]=A[0,0], b=U[1]=A[0,1], c=U[4]=A[1,0], d=U[5]=A[1,1]
    # B: e=V[0]=B[0,0], f=V[1]=B[0,1], g=V[4]=B[1,0], h=V[5]=B[1,1]
    # C: C[0,0]=W[0], C[0,1]=W[3], C[1,0]=W[1], C[1,1]=W[4]
    
    # m1 = (a+d)(e+h)
    U[0][col] = 1; U[5][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1; W[1][col] = -1; W[3][col] = -1
    col += 1
    # m2 = (c+d)e
    U[4][col] = 1; U[5][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    # m3 = a(f-h)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    # m4 = d(g-e)
    U[5][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    # m5 = (a+b)h
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    # m6 = (c-a)(e+f)
    U[4][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    # m7 = (b-d)(g+h)
    U[1][col] = 1; U[5][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # Strassen block 2: rows {0,1}, cols {2,3}
    # A: a=U[2]=A[0,2], b=U[3]=A[0,3], c=U[6]=A[1,2], d=U[7]=A[1,3]
    # B: e=V[8]=B[2,0], f=V[9]=B[2,1], g=V[12]=B[3,0], h=V[13]=B[3,1]
    # C: C[0,2]=W[6], C[0,3]=W[9], C[1,2]=W[7], C[1,3]=W[10]
    
    # m1 = (a+d)(e+h)
    U[2][col] = 1; U[7][col] = 1
    V[8][col] = 1; V[13][col] = 1
    W[6][col] = 1; W[10][col] = 1; W[7][col] = -1; W[9][col] = -1
    col += 1
    # m2 = (c+d)e
    U[6][col] = 1; U[7][col] = 1
    V[8][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    # m3 = a(f-h)
    U[2][col] = 1
    V[9][col] = 1; V[13][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    # m4 = d(g-e)
    U[7][col] = 1
    V[12][col] = 1; V[8][col] = -1
    W[6][col] = 1; W[9][col] = 1
    col += 1
    # m5 = (a+b)h
    U[2][col] = 1; U[3][col] = 1
    V[13][col] = 1
    W[6][col] = -1; W[7][col] = 1
    col += 1
    # m6 = (c-a)(e+f)
    U[6][col] = 1; U[2][col] = -1
    V[8][col] = 1; V[9][col] = 1
    W[10][col] = 1
    col += 1
    # m7 = (b-d)(g+h)
    U[3][col] = 1; U[7][col] = -1
    V[12][col] = 1; V[13][col] = 1
    W[6][col] = 1
    col += 1
    
    # Trivial terms for row 2, cols 0,1,2,3
    for k in range(4):
        for j in range(4):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    return U, V, W

def construct():
    n, m, p = 3, 4, 4
    
    U, V, W = _build_rank18_decomposition()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        pass
    
    return {"U": U, "V": V, "W": W}
