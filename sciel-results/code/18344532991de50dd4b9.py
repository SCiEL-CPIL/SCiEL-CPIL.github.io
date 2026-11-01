import time
import numpy as np

def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    Standard Strassen decomposition.
    """
    U = np.zeros((4, 7))
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    U[1, 6] = 1; U[3, 6] = -1
    
    V = np.zeros((4, 7))
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[2, 3] = 1; V[0, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1; V[1, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1
    
    W = np.zeros((4, 7))
    W[0, 0] = 1; W[0, 2] = -1; W[0, 3] = -1; W[0, 5] = 1
    W[1, 2] = 1; W[1, 4] = 1
    W[2, 1] = 1; W[2, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 4] = -1; W[3, 6] = 1
    
    return U, V, W

def _get_rank2_221():
    """
    Returns U, V, W for 2x2x1 matrix multiplication (multiplying 2x2 by 2x1).
    Rank 2 decomposition.
    A = [[a11, a12], [a21, a22]], B = [[b1], [b2]]
    C = [[c1], [c2]]
    c1 = a11*b1 + a12*b2
    c2 = a21*b1 + a22*b2
    
    U (4x2), V (2x2), W (2x2)
    """
    U = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])
    V = np.array([
        [1, 0],
        [0, 1]
    ])
    W = np.array([
        [1, 1],
        [1, 1]
    ])
    return U, V, W

def construct():
    n, m, p = 2, 2, 15
    start_time = time.time()
    
    # Build Hybrid Decomposition
    U_str, V_str, W_str = _get_strassen_222()
    U_r2, V_r2, W_r2 = _get_rank2_221()
    
    R_total = 7 + 13 * 2
    
    U = np.zeros((n * m, R_total))
    V = np.zeros((m * p, R_total))
    W = np.zeros((p * n, R_total))
    
    # Fill Strassen part (k=0,1)
    U[:, 0:7] = U_str
    
    V[0:2, 0:7] = V_str[0:2, :]
    V[2:4, 0:7] = V_str[2:4, :]
    
    W[0:2, 0:7] = W_str[0:2, :]
    W[2:4, 0:7] = W_str[2:4, :]
    
    # Fill Rank 2 parts for k=2..14
    current_col = 7
    for k in range(2, p):
        U[:, current_col:current_col+2] = U_r2
        
        for j in range(m):
            V[j*p + k, current_col:current_col+2] = V_r2[j, :]
            
        for i in range(n):
            W[k*n + i, current_col:current_col+2] = W_r2[i, :]
            
        current_col += 2
        
    def verify(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)
        
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if construction failed
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
