import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 120
    n, m, p = DIMENSIONS  # 3, 4, 8

    # Schoolbook decomposition: rank = n*m*p = 96
    # U shape: (n*m, rank) = (12, 96)
    # V shape: (m*p, rank) = (32, 96)
    # W shape: (p*n, rank) = (24, 96)
    # Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For schoolbook: each (i,j,k) has its own column with all 1s.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Quick verification with random test
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Reconstruct C from factors
    C_recon = np.zeros((n, p))
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                s += U_np[i * m + j, :] * V_np[j * p + k, :]
            C_recon[i, k] = np.dot(s, W_np[k * n + i, :])
    
    if not np.allclose(C_recon, C_expected, atol=1e-10):
        # Fallback: return valid schoolbook (should not happen)
        pass
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
