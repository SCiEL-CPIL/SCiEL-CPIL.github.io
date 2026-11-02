import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)
    
    # Schoolbook: rank = n*m*p = 48
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Should not reach here
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
