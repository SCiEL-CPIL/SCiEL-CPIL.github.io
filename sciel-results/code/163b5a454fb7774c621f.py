import numpy as np

def construct():
    n, m, p = 2, 2, 16
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2x2
    # u: (i,j) -> index i*m+j
    # v: (j,k) -> index j*p+k (local p=2)
    # w: (k,i) -> index k*n+i (local n=2)
    
    # M1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # C00, C10, C01, C11
    
    # M2
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 1, 0, -1]
    
    # M3
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, -1]
    w3 = [0, 0, 1, 1]
    
    # M4
    u4 = [0, 0, 0, 1]
    v4 = [-1, 1, 0, 0]
    w4 = [1, 1, 0, 0]
    
    # M5
    u5 = [1, 0, 1, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 0, 1, 0]
    
    # M6
    u6 = [-1, 0, 0, 1]
    v6 = [1, 0, 1, 0]
    w6 = [0, 0, 0, 0]
    
    # M7
    u7 = [0, 1, 0, -1]
    v7 = [0, 1, 0, 1]
    w7 = [-1, 0, 0, 1]
    
    ms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Verify one block
    # T_ij_k = sum_r u_ij^r v_jk^r w_ki^r
    # Check if T is identity tensor for 2x2x2
    T = np.zeros((2, 2, 2))
    for r in range(7):
        u, v, w = ms[r]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T[i,j,k] += u[i*2+j] * v[j*2+k] * w[k*2+i]
    
    # Expected T: T[i,j,k] = 1
    # Actually, the tensor for matrix mult has T[i,j,k]=1
    # If T is all 1s, it's correct.
    
    if not np.allclose(T, 1):
        # Fallback to schoolbook
        rank = 64
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Construct for 2x2x16
    # 4 blocks of 2x2x2
    for s in range(4):
        for r in range(7):
            u, v, w = ms[r]
            idx = s*7 + r
            # U
            for i in range(2):
                for j in range(2):
                    U[i*m+j, idx] = u[i*2+j]
            # V
            # Local k is 0,1. Global K = 2s + k
            for j in range(2):
                for k in range(2):
                    V[j*p + (2*s+k), idx] = v[j*2+k]
            # W
            # Local k is 0,1. Global K = 2s + k
            for k in range(2):
                for i in range(2):
                    W[(2*s+k)*n + i, idx] = w[k*2+i]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
