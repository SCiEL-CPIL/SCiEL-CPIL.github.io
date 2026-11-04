"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Schoolbook decomposition: rank = n*m*p = 40
    rank = n * m * p
    
    # U has shape (n*m, rank) = (8, 40)
    # V has shape (m*p, rank) = (20, 40)  
    # W has shape (p*n, rank) = (10, 40)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Verify the decomposition is correct
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Test with a few random matrices
    rng = np.random.RandomState(42)
    for trial in range(3):
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Should never happen for schoolbook, but just in case
            raise RuntimeError("Schoolbook verification failed")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
