import numpy as np

def construct():
    n, m, p = 2, 3, 4
    r = 8  # Theoretical minimum rank
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.rand(n * m, r) * 0.1
    V = np.random.rand(m * p, r) * 0.1
    W = np.random.rand(p * n, r) * 0.1
    
    # ALS Iterations
    for _ in range(1000):
        # Update U
        for i in range(n):
            for j in range(m):
                # A_{i,j} is p x r
                A = np.zeros((p, r))
                for k in range(p):
                    for c in range(r):
                        A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                # Solve A * U_vec = ones(p)
                U_vec, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[i * m + j, :] = U_vec
        
        # Update V
        for j in range(m):
            for k in range(p):
                # B_{j,k} is n x r
                B = np.zeros((n, r))
                for i in range(n):
                    for c in range(r):
                        B[i, c] = U[i * m + j, c] * W[k * n + i, c]
                # Solve B * V_vec = ones(n)
                V_vec, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                V[j * p + k, :] = V_vec
        
        # Update W
        for k in range(p):
            for i in range(n):
                # C_{k,i} is m x r
                C = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        C[j, c] = U[i * m + j, c] * V[j * p + k, c]
                # Solve C * W_vec = ones(m)
                W_vec, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                W[k * n + i, :] = W_vec
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err += (s - 1.0) ** 2
        if err < 1e-10:
            break
    
    # Round to nearest integer or half-integer if close
    def clean_matrix(M):
        M_clean = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                # Check if close to integer or half-integer
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-6:
                    M_clean[i, j] = nearest_int
                elif abs(val - nearest_half) < 1e-6:
                    M_clean[i, j] = nearest_half
                else:
                    M_clean[i, j] = val
        return M_clean

    U = clean_matrix(U)
    V = clean_matrix(V)
    W = clean_matrix(W)
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
