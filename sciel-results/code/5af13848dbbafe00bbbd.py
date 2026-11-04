import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(i, k, j, coeff=1.0):
        """Add a rank-1 term: coeff * A[i,k] * B[k,j] contributes to C[i,j]"""
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        u_idx = i * m + k
        v_idx = k * p + j
        w_idx = j * n + i
        
        u_vec[u_idx] = coeff
        v_vec[v_idx] = 1.0
        w_vec[w_idx] = 1.0
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Schoolbook decomposition: one rank-1 term per (i,k,j)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                if time.time() - _START > _DEADLINE - 10:
                    break
                add_rank1(i, k, j, 1.0)
            if time.time() - _START > _DEADLINE - 10:
                break
        if time.time() - _START > _DEADLINE - 10:
            break
    
    rank = len(U_cols)
    
    # Ensure we have at least one term
    if rank == 0:
        add_rank1(0, 0, 0, 1.0)
        rank = 1
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
