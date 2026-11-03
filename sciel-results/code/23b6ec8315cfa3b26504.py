"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((dim_u, rank_sb))
    V_sb = np.zeros((dim_v, rank_sb))
    W_sb = np.zeros((dim_w, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        if r != V.shape[1] or r != W.shape[1]:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def round_to_half_integers(mat):
        """Round to nearest half-integer and verify."""
        return np.round(mat * 2) / 2
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(dim_u, dim_v, dim_w) = 20
    lower_bound = max(dim_u, dim_v, dim_w)
    
    for rank in range(lower_bound, rank_sb):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with ALS
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 1000 + attempt)
            
            # Random initialization with small values
            U = np.random.randn(dim_u, rank) * 0.5
            V = np.random.randn(dim_v, rank) * 0.5
            W = np.random.randn(dim_w, rank) * 0.5
            
            # ALS iterations
            converged = False
            for als_iter in range(100):
                if time.time() > deadline:
                    break
                
                # Solve for U given V, W
                # For each (i,j): U[i*m+j, :] must satisfy
                # sum_k W[k*n+i, :] * V[j*p+k, :] * U[i*m+j, :] = 1 for all k
                # This is: (V[j*p:(j+1)*p, :] * W.T) @ U[i*m+j, :].T = ones(p)
                # i.e., A @ U_row.T = ones(p) where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                
                for i in range(n):
                    for j in range(m):
                        idx_u = i * m + j
                        # Build A: A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        A = V[j*p:(j+1)*p, :] * W.T  # shape (p, rank)
                        # Solve A @ U_row.T = ones(p)
                        # Use least squares
                        U_row, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[idx_u, :] = U_row
                
                # Solve for V given U, W
                for j in range(m):
                    for k in range(p):
                        idx_v = j * p + k
                        # Build A: A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        # Solve A @ V_row.T = ones(n)
                        V_row, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[idx_v, :] = V_row
                
                # Solve for W given U, V
                for k in range(p):
                    for i in range(n):
                        idx_w = k * n + i
                        # Build A: A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        # Solve A @ W_row.T = ones(m)
                        W_row, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[idx_w, :] = W_row
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            s = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-8:
                    converged = True
                    break
            
            if converged:
                # Round to half-integers
                U_r = round_to_half_integers(U)
                V_r = round_to_half_integers(V)
                W_r = round_to_half_integers(W)
                
                if verify(U_r, V_r, W_r):
                    best_result = (U_r, V_r, W_r)
                    best_rank = rank
                    break
        
        if best_rank == rank:
            break  # Found the best possible for this rank
    
    # Convert to lists
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
