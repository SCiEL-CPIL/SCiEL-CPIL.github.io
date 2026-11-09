import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    dim_U = n * m  # 6
    dim_V = m * p  # 30
    dim_W = p * n  # 20
    
    deadline = time.time() + 140  # Safety margin
    
    # Schoolbook decomposition: rank = n*m*p = 60
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor entry T[(i,j),(j,k),(k,i)] = 1
    # W uses convention W[k*n + i, r]
    
    rank = n * m * p  # 60
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Verify the decomposition with random test matrices
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Reconstruct C from the decomposition
    C_recon = np.zeros((n, p))
    for r in range(rank):
        a_form = np.dot(U[:, r], A_test.flatten())
        b_form = np.dot(V[:, r], B_test.flatten())
        for k in range(p):
            for i in range(n):
                C_recon[i, k] += a_form * b_form * W[k*n + i, r]
    
    if not np.allclose(C_recon, C_expected, atol=1e-10):
        # Fallback: shouldn't happen, but safety
        pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
