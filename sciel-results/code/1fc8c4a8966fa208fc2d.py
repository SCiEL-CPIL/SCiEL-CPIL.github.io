"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline (rank = n*m*p = 125)
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook
    
    # Attempt to find a lower-rank decomposition using a randomized approach
    # For 5x5x5 matrix multiplication, the known optimal rank is 19 (Strassen-like variants)
    # but finding it algorithmically is very hard. We'll try a simple heuristic.
    
    # Try a few random low-rank attempts to see if we can beat schoolbook
    # This is unlikely to work for 5x5x5 in reasonable time, but let's try briefly
    
    for attempt_rank in [20, 25, 30]:
        if time.time() > deadline - 10:
            break
        
        # Generate random factor matrices
        np.random.seed(attempt_rank)  # Deterministic for reproducibility
        U_try = np.random.randn(n * m, attempt_rank)
        V_try = np.random.randn(m * p, attempt_rank)
        W_try = np.random.randn(p * n, attempt_rank)
        
        # Check if this decomposition works: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # This is equivalent to checking if the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # equals 1 for all i,j,k
        
        # Build the tensor from the factors
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We can check this by constructing the tensor and comparing to ones
        
        T_reconstructed = np.zeros((n, m, p), dtype=np.float64)
        for r in range(attempt_rank):
            u_col = U_try[:, r]  # shape (n*m,)
            v_col = V_try[:, r]  # shape (m*p,)
            w_col = W_try[:, r]  # shape (p*n,)
            
            # Reshape u_col to (n, m), v_col to (m, p), w_col to (p, n)
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            
            # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_reconstructed[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        
        # Check if T_reconstructed is all ones
        if np.allclose(T_reconstructed, np.ones((n, m, p)), atol=1e-10):
            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
            best_rank = attempt_rank
            break
    
    # Return the best decomposition found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
