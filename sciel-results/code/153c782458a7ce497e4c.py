"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # The tensor T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # We want rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Start with schoolbook rank and try to reduce
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Build the full tensor as a 3D array
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Try ranks from low to high
    # Known lower bounds: rank >= max(n*m, m*p, p*n) is not right.
    # For 2x3x8, let's try ranks from 8 upward
    for rank in range(8, n * m * p + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        n_tries = 20
        for trial in range(n_tries):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(42 + trial * 1000 + rank)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Iterative refinement: alternate between fixing two factors and solving for the third
            converged = False
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor estimate
                # T_est[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # Update U: for each (i,j), solve least squares
                # T[i,j,:] = U[i*m+j,:] @ (V_j * W_i)^T where V_j = V[j,:], W_i = W[:,i]
                # Actually: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For fixed V, W: T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
                # So for each (i,j): T[i,j,:] = U[i*m+j,:] @ M_ij^T where M_ij[c,k] = V[j*p+k,c]*W[k*n+i,c]
                
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # M_ij is (rank, p)
                        M_ij = np.zeros((rank, p))
                        for k in range(p):
                            M_ij[:, k] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve: T[i,j,:] = U[i*m+j,:] @ M_ij^T
                        # i.e., T[i,j,:] @ M_ij = U[i*m+j,:] @ M_ij^T @ M_ij
                        # Use least squares: U[i*m+j,:] = T[i,j,:] @ pinv(M_ij^T)
                        try:
                            new_U[i * m + j, :] = np.linalg.lstsq(M_ij.T, T[i, j, :], rcond=None)[0]
                        except:
                            break
                U = new_U
                
                # Update V: for each (j,k)
                new_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # T[:,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # For fixed U, W: T[i,j,k] = sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c])
                        # So for each (j,k): T[:,j,k] = V[j*p+k,:] @ N_jk^T where N_jk[c,i] = U[i*m+j,c]*W[k*n+i,c]
                        N_jk = np.zeros((rank, n))
                        for i in range(n):
                            N_jk[:, i] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            new_V[j * p + k, :] = np.linalg.lstsq(N_jk.T, T[:, j, k], rcond=None)[0]
                        except:
                            break
                V = new_V
                
                # Update W: for each (k,i)
                new_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # T[i,:,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # For fixed U, V: T[i,j,k] = sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c])
                        # So for each (k,i): T[:,j,k]... wait, T[i,j,k] with i,j varying
                        # T[i,j,k] = sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c])
                        # For fixed (k,i): the vector over j is T[i,:,k] = W[k*n+i,:] @ P_ki^T
                        # where P_ki[c,j] = U[i*m+j,c] * V[j*p+k,c]
                        P_ki = np.zeros((rank, m))
                        for j in range(m):
                            P_ki[:, j] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            new_W[k * n + i, :] = np.linalg.lstsq(P_ki.T, T[i, :, k], rcond=None)[0]
                        except:
                            break
                W = new_W
                
                # Check convergence
                # Compute residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            residual += (val - 1.0) ** 2
                if residual < 1e-10:
                    converged = True
                    break
                
                if time.time() > deadline - 10:
                    break
            
            if converged:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank < n * m * p:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
