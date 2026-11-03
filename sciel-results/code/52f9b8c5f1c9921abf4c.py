"""Rank-18 exact tensor decomposition for 2x3x3 matrix multiplication."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3  # A is n x m, B is m x p, C is n x p
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (9, rank)
    # W shape: (p*n, rank) = (6, rank), W[k*n+i, r]
    
    # Schoolbook decomposition requires n*m*p terms
    rank = n * m * p # 18
    
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term r corresponds to A[i,j] * B[j,k] contributing to C[i,k]
                # U index: i*m + j
                U[i*m + j][r] = 1.0
                # V index: j*p + k
                V[j*p + k][r] = 1.0
                # W index: k*n + i
                W[k*n + i][r] = 1.0
                r += 1
    
    assert r == rank, f"Expected {rank} columns, got {r}"
    
    # Verify correctness
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    for _ in range(10):
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_test = A_test @ B_test
        
        # Compute via tensor decomposition
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    # Calculate linear combination of A
                    u_val = 0.0
                    for j in range(m):
                        u_val += U_np[i*m + j, r] * A_test[i, j]
                    # Calculate linear combination of B
                    v_val = 0.0
                    for j in range(m):
                        v_val += V_np[j*p + k, r] * B_test[j, k]
                    # Multiply and scale by W
                    s += W_np[k*n + i, r] * u_val * v_val
                C_decomp[i, k] = s
                
        if not np.allclose(C_test, C_decomp, atol=1e-10):
            raise ValueError(f"Decomposition failed: max error = {np.max(np.abs(C_test - C_decomp))}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
