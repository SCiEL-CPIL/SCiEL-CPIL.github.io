import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Try the deterministic rank-49 Strassen decomposition
    U, V, W = _build_strassen_4x4()
    
    if time.time() > deadline:
        return _schoolbook()

    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 2. If deterministic fails (should not happen), try a different structural approach:
    #    Randomized initialization + Gradient Descent (or just return schoolbook if time is short)
    #    Given the 120s limit and the difficulty of finding rank < 49, we fall back to schoolbook 
    #    if the deterministic one fails, but we ensure the deterministic one is correct.
    
    # For the purpose of this task, the deterministic one MUST work.
    # If it doesn't, it's a bug in my implementation of the deterministic one.
    # I will re-verify the indices in _build_strassen_4x4.
    
    # Fallback to schoolbook rank 64
    return _schoolbook()

def _build_strassen_4x4():
    """
    Build rank-49 Strassen decomposition for 4x4 MM.
    """
    a2, b2, c2 = _strassen_2x2()
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for s1 in range(7):
        for s2 in range(7):
            col = 7 * s1 + s2
            
            # U: index (i, j)
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U[i * 4 + j, col] = a2[s1, i1, j1] * a2[s2, i2, j2]
            
            # V: index (j, k)
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            V[j * 4 + k, col] = b2[s1, j1, k1] * b2[s2, j2, k2]
            
            # W: index (k, i)
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            k = 2 * k1 + k2
                            W[k * 4 + i, col] = c2[i1, k1, s1] * c2[i2, k2, s2]
    
    return U, V, W

def _strassen_2x2():
    a = np.zeros((7, 2, 2), dtype=int)
    b = np.zeros((7, 2, 2), dtype=int)
    c = np.zeros((2, 2, 7), dtype=int)
    
    a[0] = [[1, 0], [0, 1]]
    b[0] = [[1, 0], [0, 1]]
    
    a[1] = [[0, 0], [1, 1]]
    b[1] = [[1, 0], [0, 0]]
    
    a[2] = [[1, 0], [0, 0]]
    b[2] = [[0, 0], [1, -1]]
    
    a[3] = [[0, 0], [0, 1]]
    b[3] = [[-1, 1], [0, 0]]
    
    a[4] = [[1, 1], [0, 0]]
    b[4] = [[0, 0], [0, 1]]
    
    a[5] = [[-1, 0], [1, 0]]
    b[5] = [[1, 1], [0, 0]]
    
    a[6] = [[0, 1], [0, -1]]
    b[6] = [[0, 0], [1, 1]]
    
    c[0, 0] = [1, 0, 0, 1, -1, 0, 1]
    c[0, 1] = [0, 0, 1, 0, 1, 0, 0]
    c[1, 0] = [0, 1, 0, 1, 0, 0, 0]
    c[1, 1] = [1, -1, 1, 0, 0, 1, 0]
    
    return a, b, c

def _verify(U, V, W, tol=1e-9):
    n, m, p = 4, 4, 4
    rank = U.shape[1]
    U4 = np.asarray(U).reshape(n, m, rank)
    V4 = np.asarray(V).reshape(m, p, rank)
    W4 = np.asarray(W).reshape(p, n, rank)
    
    # Use einsum for robustness
    # recon[i, j, k] = sum_r U4[i, j, r] * V4[j, k, r] * W4[k, i, r]
    recon = np.einsum('ijr,jkr,kir->ijk', U4, V4, W4)
    
    target = np.ones((n, m, p))
    return np.allclose(recon, target, atol=tol)

def _schoolbook():
    n, m, p = 4, 4, 4
    rank = 64
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
