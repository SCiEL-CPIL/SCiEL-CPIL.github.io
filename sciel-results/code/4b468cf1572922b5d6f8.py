import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
    
    num_eqs = len(indices)
    w_dim = p * n
    
    def find_decomposition(R, max_attempts=100):
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random Integer U, V
            U = np.random.randint(-2, 3, size=(n * m, R))
            V = np.random.randint(-2, 3, size=(m * p, R))
            
            A = np.zeros((num_eqs, R * w_dim))
            T_vec = np.ones(num_eqs)
            
            for idx, (u_idx, v_idx, w_idx) in enumerate(indices):
                for r in range(R):
                    A[idx, r * w_dim + w_idx] = U[u_idx, r] * V[v_idx, r]
            
            try:
                W_vec, residuals, rank, sv = np.linalg.lstsq(A, T_vec, rcond=None)
                if residuals[0] < 1e-9:
                    W = W_vec.reshape(R, w_dim).T
                    # Check if W is close to half-integers
                    W_check = np.round(W * 2) / 2
                    if np.allclose(W, W_check, atol=1e-9):
                        return U, V, W_check
            except np.linalg.LinAlgError:
                continue
        
        return None

    schoolbook_rank = n * m * p
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school

    for R in range(schoolbook_rank - 1, 0, -1):
        if time.time() > deadline:
            break
        
        attempts = 100
        result = find_decomposition(R, attempts)
        if result is not None:
            best_U, best_V, best_W = result
            break # Found a valid one, assume it's good enough for now?
            # Or continue to find lower?
            # Let's break to save time.

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
