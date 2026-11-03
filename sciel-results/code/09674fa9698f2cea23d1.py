"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Schoolbook decomposition: rank = n*m*p = 90
    # For each (i,j,k) triple, we have one term
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    rank = n * m * p  # 90
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, c] = 1.0
                V[v_idx, c] = 1.0
                W[w_idx, c] = 1.0
                c += 1
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p, seed=42):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rng = np.random.RandomState(seed)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        rank = U_mat.shape[1]
        C_computed = np.zeros((n, p), dtype=float)
        
        for c in range(rank):
            u_col = U_mat[:, c].reshape(n, m)
            v_col = V_mat[:, c].reshape(m, p)
            w_col = W_mat[:, c].reshape(p, n)
            
            # C[i,k] += W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
            for i in range(n):
                alpha = np.dot(u_col[i, :], A[i, :])
                for k in range(p):
                    beta = np.dot(v_col[:, k], B[:, k])
                    C_computed[i, k] += w_col[k, i] * alpha * beta
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Verify schoolbook
    if verify(U, V, W, n, m, p):
        # Convert to lists for the return format
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If verification failed (shouldn't happen), still return the schoolbook
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
