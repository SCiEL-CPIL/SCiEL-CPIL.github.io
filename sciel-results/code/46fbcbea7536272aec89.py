import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _verify_decomposition(U, V, W, n, m, p):
    T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
    return np.allclose(T_recon, T_target, atol=1e-9)

def _strassen_222():
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, _ = V1.shape
    m2, _ = V2.shape
    p1, _ = W1.shape
    p2, _ = W2.shape
    
    R = r1 * r2
    U = np.zeros((n1 * n2, R), dtype=float)
    V = np.zeros((m1 * m2, R), dtype=float)
    W = np.zeros((p1 * p2, R), dtype=float)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            
            U[:, r_idx] = np.kron(U1[:, r1_idx], U2[:, r2_idx])
            V[:, r_idx] = np.kron(V1[:, r1_idx], V2[:, r2_idx])
            W[:, r_idx] = np.kron(W1[:, r1_idx], W2[:, r2_idx])
            
    return U, V, W

def _direct_sum_decomposition(U1, V1, W1):
    n, r = U1.shape
    m, _ = V1.shape
    p, _ = W1.shape
    
    R = 2 * r
    U = np.zeros((n, R), dtype=float)
    V = np.zeros((2 * m, R), dtype=float)
    W = np.zeros((2 * p, R), dtype=float)
    
    U[:, :r] = U1
    V[:m, :r] = V1
    W[:p, :r] = W1
    
    U[:, r:] = U1
    V[m:, r:] = V1
    W[p:, r:] = W1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    U2, V2, W2 = _strassen_222()
    U444, V444, W444 = _kron_decomposition(U2, V2, W2, U2, V2, W2)
    U, V, W = _direct_sum_decomposition(U444, V444, W444)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        schoolbook_rank = n * m * p
        U_school = np.zeros((n*m, schoolbook_rank))
        V_school = np.zeros((m*p, schoolbook_rank))
        W_school = np.zeros((p*n, schoolbook_rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_school[ii*m + jj, col] = 1
                    V_school[jj*p + kk, col] = 1
                    W_school[kk*n + ii, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
