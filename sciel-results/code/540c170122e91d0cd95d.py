import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Rank 19 decomposition
    rank = 19
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    r = 0
    
    # Part 1: j=0
    # A[i,0] * B[0,k]
    for i in range(n):
        U[i*m + 0][r] = 1
    for k in range(p):
        V[0*p + k][r] = 1
    for k in range(p):
        for i in range(n):
            W[k*n + i][r] = 1
    r += 1
    
    # Part 2: j in {1, 2}
    # Split k into {0,1}, {2,3}, {4}
    
    def add_strassen_block(j_offset, k_offset):
        nonlocal r
        # j_offset is 1, k_offset is 0 or 2
        # Local indices: i in 0..1, jl in 0..1, kl in 0..1
        # Global: j = j_offset + jl, k = k_offset + kl
        
        # M1 = (A00+A11)(B00+B11)
        U[(0)*m + (j_offset+0)][r] = 1
        U[(1)*m + (j_offset+1)][r] = 1
        V[(j_offset+0)*p + (k_offset+0)][r] = 1
        V[(j_offset+1)*p + (k_offset+1)][r] = 1
        W[(k_offset+0)*n + 0][r] = 1
        W[(k_offset+1)*n + 1][r] = 1
        r += 1
        
        # M2 = (A10+A11)B00
        U[(1)*m + (j_offset+0)][r] = 1
        U[(1)*m + (j_offset+1)][r] = 1
        V[(j_offset+0)*p + (k_offset+0)][r] = 1
        W[(k_offset+0)*n + 1][r] = 1
        W[(k_offset+1)*n + 1][r] = -1
        r += 1
        
        # M3 = A00(B01-B11)
        U[(0)*m + (j_offset+0)][r] = 1
        V[(j_offset+0)*p + (k_offset+1)][r] = 1
        V[(j_offset+1)*p + (k_offset+1)][r] = -1
        W[(k_offset+1)*n + 0][r] = 1
        W[(k_offset+1)*n + 1][r] = 1
        r += 1
        
        # M4 = A11(B10-B00)
        U[(1)*m + (j_offset+1)][r] = 1
        V[(j_offset+1)*p + (k_offset+0)][r] = 1
        V[(j_offset+0)*p + (k_offset+0)][r] = -1
        W[(k_offset+0)*n + 0][r] = 1
        W[(k_offset+0)*n + 1][r] = 1
        r += 1
        
        # M5 = (A00+A01)B11
        U[(0)*m + (j_offset+0)][r] = 1
        U[(0)*m + (j_offset+1)][r] = 1
        V[(j_offset+1)*p + (k_offset+1)][r] = 1
        W[(k_offset+0)*n + 0][r] = -1
        W[(k_offset+1)*n + 0][r] = 1
        r += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(1)*m + (j_offset+0)][r] = 1
        U[(0)*m + (j_offset+0)][r] = -1
        V[(j_offset+0)*p + (k_offset+0)][r] = 1
        V[(j_offset+0)*p + (k_offset+1)][r] = 1
        W[(k_offset+1)*n + 1][r] = 1
        r += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[(0)*m + (j_offset+1)][r] = 1
        U[(1)*m + (j_offset+1)][r] = -1
        V[(j_offset+1)*p + (k_offset+0)][r] = 1
        V[(j_offset+1)*p + (k_offset+1)][r] = 1
        W[(k_offset+0)*n + 0][r] = 1
        r += 1

    add_strassen_block(1, 0) # k=0,1
    add_strassen_block(1, 2) # k=2,3
    
    # Part 2c: k=4
    # Schoolbook for 2x2 * 2x1
    # C[i,4] = A[i,1]B[1,4] + A[i,2]B[2,4]
    for i in range(n):
        for j_local in range(2):
            j = 1 + j_local
            U[i*m + j][r] = 1
            V[j*p + 4][r] = 1
            W[4*n + i][r] = 1
            r += 1
            
    # Verify
    def verify(U_list, V_list, W_list, r):
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        # Construct T
        # T[i, j, k] should be 1
        # Check sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
        
        # Flatten indices
        # U: n*m x r
        # V: m*p x r
        # W: p*n x r
        
        # We can check by picking random A, B
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        U_flat = U_arr.T @ A.flatten() # r x 1
        V_flat = V_arr.T @ B.flatten() # r x 1
        P = U_flat * V_flat # r x 1
        C_flat = W_arr @ P # p*n x 1
        C_calc = C_flat.reshape(p, n).T # n x p
        
        return np.allclose(C_calc, C_true, atol=1e-9)

    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
