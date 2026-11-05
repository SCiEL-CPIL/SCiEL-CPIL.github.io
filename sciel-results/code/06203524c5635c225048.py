import numpy as np
import random
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank_school = n * m * p
    
    # Helper to check if a set of factors is a valid exact decomposition
    def is_valid_exact(U, V, W):
        # U, V, W are numpy arrays of shape (nm, R), (mp, R), (pn, R)
        # Check integer/half-integer
        U2 = 2 * U
        V2 = 2 * V
        W2 = 2 * W
        
        # Check if close to integers
        if not np.allclose(U2, np.round(U2), atol=1e-6):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-6):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-6):
            return False
            
        # Check exact decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    prod = U[idx_u] * V[idx_v] * W[idx_w]
                    s = np.sum(prod)
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Helper to construct schoolbook decomposition
    def schoolbook():
        R = rank_school
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Compute the tensor values T[i,j,k] = 1 for all i,j,k
    # We'll use IHT to find a sparse decomposition
    
    # Start time
    start_time = time.time()
    time_limit = 110.0
    
    # Try to find lower rank using IHT approach
    # The idea: for each candidate rank R, we try to find U, V, W with many zeros
    # such that the tensor is exactly reconstructed
    
    # We'll use a greedy approach: start with full rank and try to eliminate terms
    
    def compute_tensor_error(U, V, W, R):
        """Compute the error in reconstructing the tensor"""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                    err += (1.0 - val) ** 2
        return err
    
    def iht_solve(R, max_iter=100, tol=1e-8):
        """
        Use Iterative Hard Thresholding to find a sparse decomposition of rank R.
        Returns (U, V, W) if found, else None.
        """
        # Initialize with random values
        U = np.random.randn(nm, R) * 0.1
        V = np.random.randn(mp, R) * 0.1
        W = np.random.randn(pn, R) * 0.1
        
        # We want to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
        # where U_r is the r-th column, etc.
        
        # Use alternating minimization with sparsity constraints
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit:
                return None
            
            # Update U: for each row i*m+j, find best values
            for idx_u in range(nm):
                i = idx_u // m
                j = idx_u % m
                # For this (i,j), we need: sum_r U[idx_u, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is a linear system in U[idx_u, :]
                # For each k, we have: sum_r U[idx_u, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                # Let A_k[r] = V[j*p+k, r] * W[k*n+i, r]
                # Then A @ U[idx_u, :] = ones(p)
                
                A = np.zeros((p, R))
                for k in range(p):
                    A[k] = V[j * p + k] * W[k * n + i]
                
                # Solve A @ x = ones(p) in least squares sense
                # But we also want x to be sparse
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                except:
                    continue
                
                # Apply hard thresholding: keep only top R values (but we have R columns already)
                # Actually, we want to encourage sparsity in the overall decomposition
                # For now, just use the least squares solution
                U[idx_u] = x
            
            # Update V: for each row j*p+k, find best values
            for idx_v in range(mp):
                j = idx_v // p
                k = idx_v % p
                # For this (j,k), we need: sum_r U[i*m+j, r] * V[idx_v, r] * W[k*n+i, r] = 1 for all i
                # Let B_i[r] = U[i*m+j, r] * W[k*n+i, r]
                # Then B @ V[idx_v, :] = ones(n)
                
                B = np.zeros((n, R))
                for i in range(n):
                    B[i] = U[i * m + j] * W[k * n + i]
                
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                except:
                    continue
                
                V[idx_v] = x
            
            # Update W: for each row k*n+i, find best values
            for idx_w in range(pn):
                k = idx_w // n
                i = idx_w % n
                # For this (k,i), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[idx_w, r] = 1 for all j
                # Let C_j[r] = U[i*m+j, r] * V[j*p+k, r]
                # Then C @ W[idx_w, :] = ones(m)
                
                C = np.zeros((m, R))
                for j in range(m):
                    C[j] = U[i * m + j] * V[j * p + k]
                
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                except:
                    continue
                
                W[idx_w] = x
            
            # Check convergence
            err = compute_tensor_error(U, V, W, R)
            if err < tol:
                # Round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                if is_valid_exact(U_round, V_round, W_round):
                    return (U_round, V_round, W_round)
        
        return None
    
    # Try ranks from 47 down to 35
    best_result = None
    for R in range(rank_school - 1, max(rank_school - 15, 30), -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
            
            result = iht_solve(R)
            if result is not None:
                U, V, W = result
                best_result = (R, U, V, W)
                break
        
        if best_result:
            break
    
    if best_result:
        R, U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
