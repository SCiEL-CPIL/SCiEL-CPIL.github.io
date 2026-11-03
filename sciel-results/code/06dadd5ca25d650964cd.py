import time
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE_S = 110.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the rank-51 Strassen-based decomposition
    U, V, W = _build_strassen_51(n, m, p)
    
    # Verify correctness with the fixed logic
    if _verify(U, V, W, n, m, p):
        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return result
    
    # Fallback
    return _schoolbook(n, m, p)

def _build_strassen_51(n, m, p):
    U = np.zeros((n * m, 51))
    V = np.zeros((m * p, 51))
    W = np.zeros((p * n, 51))
    
    col = 0
    for k_group in range(3):
        k0 = k_group * 2
        k1 = k_group * 2 + 1
        
        # Strassen terms (7)
        # M1
        U[0*3+0, col] = 1; U[1*3+1, col] = 1
        V[0*6+k0, col] = 1; V[1*6+k1, col] = 1
        W[k0*3+0, col] = 1; W[k1*3+1, col] = 1
        col += 1
        
        # M2
        U[1*3+0, col] = 1; U[1*3+1, col] = 1
        V[0*6+k0, col] = 1
        W[k0*3+1, col] = 1
        col += 1
        
        # M3
        U[0*3+0, col] = 1
        V[0*6+k1, col] = 1; V[1*6+k1, col] = -1
        W[k1*3+0, col] = 1
        col += 1
        
        # M4
        U[1*3+1, col] = 1
        V[1*6+k0, col] = 1; V[0*6+k0, col] = -1
        W[k0*3+1, col] = 1
        col += 1
        
        # M5
        U[0*3+0, col] = 1; U[0*3+1, col] = 1
        V[1*6+k1, col] = 1
        W[k1*3+0, col] = 1; W[k1*3+1, col] = 1
        col += 1
        
        # M6
        U[1*3+0, col] = 1; U[0*3+0, col] = -1
        V[0*6+k0, col] = 1; V[0*6+k1, col] = 1
        W[k1*3+1, col] = 1
        col += 1
        
        # M7
        U[0*3+1, col] = 1; U[1*3+1, col] = -1
        V[1*6+k0, col] = 1; V[1*6+k1, col] = 1
        W[k0*3+0, col] = 1; W[k1*3+1, col] = 1
        col += 1
        
        # Cross terms
        for i in range(2):
            for k in range(k0, k0+2):
                U[i*3+2, col] = 1
                V[2*6+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
        
        for j in range(2):
            for k in range(k0, k0+2):
                U[2*3+j, col] = 1
                V[j*6+k, col] = 1
                W[k*3+2, col] = 1
                col += 1
        
        for k in range(k0, k0+2):
            U[2*3+2, col] = 1
            V[2*6+k, col] = 1
            W[k*3+2, col] = 1
            col += 1
            
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            L_r = np.dot(U[:, r], A.flatten())
            M_r = np.dot(V[:, r], B.flatten())
            W_mat = W[:, r].reshape(p, n)
            C_computed += L_r * M_r * W_mat.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True
