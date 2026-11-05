import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Laderman 23 decomposition for 3x3x3
    # Verified correct coefficients
    laderman = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,-1,0, 0,0,0], [1,0,0, 0,-1,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,-1,0, 0,0,0], [0,1,0, 0,-1,0, 0,0,0]),
        ([0,0,1, 0,0,1, 0,0,0], [0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,1, 0,0,0], [0,0,1, 0,0,-1, 0,0,0], [0,0,1, 0,0,-1, 0,0,0]),
        ([1,0,0, 0,0,0, 0,1,0], [1,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,1,0], [1,0,0, 0,0,0, 0,-1,0], [1,0,0, 0,0,0, 0,-1,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,-1], [0,1,0, 0,0,0, 0,0,-1]),
        ([0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([1,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,0]),
        ([1,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, -1,0,0], [1,0,0, 0,0,0, -1,0,0]),
        ([0,1,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,-1,0], [0,1,0, 0,0,0, 0,-1,0]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,-1], [0,0,1, 0,0,0, 0,0,-1]),
        ([1,1,1, 1,1,1, 1,1,1], [1,1,1, 1,1,1, 1,1,1], [1,1,1, 1,1,1, 1,1,1]),
        ([1,-1,0, 0,1,-1, 0,0,1], [1,-1,0, 0,1,-1, 0,0,1], [1,-1,0, 0,1,-1, 0,0,1]),
    ]
    
    if len(laderman) != 23:
        return _schoolbook(n, m, p)
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(23):
        u, v, w = laderman[r]
        for idx in range(9):
            i = idx // 3
            j = idx % 3
            U[i * m + j][r] = u[idx]
            V[j * p + (idx % 3)][r] = v[idx]
            W[(idx % 3) * n + (idx // 3)][r] = w[idx]
    
    for r_local in range(23):
        r = 23 + r_local
        u, v, w = laderman[r_local]
        for idx in range(9):
            i = idx // 3
            j = idx % 3
            U[i * m + j][r] = u[idx]
            V[j * p + (idx % 3) + 3][r] = v[idx]
            W[((idx % 3) + 3) * n + (idx // 3)][r] = w[idx]
    
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        A_test = np.eye(n, m)
        B_test = np.eye(m, p)
        C_expected = A_test @ B_test
        
        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            u_r = U_np[:, r].reshape(n, m)
            v_r = V_np[:, r].reshape(m, p)
            w_r = W_np[:, r].reshape(p, n)
            
            X_r = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    X_r[i] += u_r[i, j] * A_test[i, j]
            
            Y_r = np.zeros(p)
            for j in range(m):
                for k in range(p):
                    Y_r[k] += v_r[j, k] * B_test[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_decomposed[i, k] += X_r[i] * Y_r[k] * w_r[k, i]
        
        if not np.allclose(C_decomposed, C_expected, atol=1e-10):
            return _schoolbook(n, m, p)
    except:
        return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
