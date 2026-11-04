import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def get_rank59():
        """
        Rank-59 decomposition using Strassen on a 2x2x2 sub-block.
        Saves 1 term compared to schoolbook (60 - 1 = 59).
        """
        rank = 59
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)
        
        col = 0
        # Schoolbook terms for all (i,j,k) except j in {0,1} and k in {0,1}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not (j < 2 and k < 2):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
        
        # Strassen terms for j in {0,1}, k in {0,1}
        # M1 = (A00 + A11)(B00 + B11)
        U[0, col] = 1; U[6, col] = 1
        V[0, col] = 1; V[7, col] = 1
        W[0, col] = 1; W[3, col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[5, col] = 1; U[6, col] = 1
        V[0, col] = 1
        W[2, col] = 1
        col += 1
        
        # M3 = A00(B10 - B11)
        U[0, col] = 1
        V[6, col] = 1; V[7, col] = -1
        W[1, col] = 1
        col += 1
        
        # M4 = A11(B01 - B00)
        U[6, col] = 1
        V[1, col] = 1; V[0, col] = -1
        W[0, col] = 1; W[2, col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[0, col] = 1; U[1, col] = 1
        V[7, col] = 1
        W[0, col] = -1; W[1, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[5, col] = 1; U[0, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[3, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1, col] = 1; U[6, col] = -1
        V[6, col] = 1; V[7, col] = 1
        W[0, col] = 1
        col += 1
        
        assert col == rank, f"Expected {rank}, got {col}"
        return U, V, W
    
    idx_u = np.zeros(n * m * p, dtype=np.int32)
    idx_v = np.zeros(n * m * p, dtype=np.int32)
    idx_w = np.zeros(n * m * p, dtype=np.int32)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u[idx] = i * m + j
                idx_v[idx] = j * p + k
                idx_w[idx] = k * n + i
                idx += 1
    
    def check(U, V, W):
        vals = U[idx_u] * V[idx_v] * W[idx_w]
        return np.all(vals.sum(axis=1) == 1.0)
    
    U, V, W = get_rank59()
    if check(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    rank = 60
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
