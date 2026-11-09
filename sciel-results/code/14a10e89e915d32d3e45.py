import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # Tensor T is defined by T[i, j, k] = 1 if i==j==k? No.
    # Matrix multiplication: C = A @ B
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor index: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Wait, the trace convention W[k*n+i, r] matches C[i, k] index.
    # U index: i*m + j. A[i, j].
    # V index: j*p + k. B[j, k].
    # W index: k*n + i. C[i, k].
    
    # The tensor T is a 3-index object T[i, j, k] such that 
    # <T, A \otimes B \otimes C> = sum A[i, j] B[j, k] C[k, i] = trace(C A B) ?
    # Actually, standard tensor decomposition for MM:
    # T_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in C_{k, i}.
    # C_{k, i} = sum_j A_{i, j} B_{j, k}.
    # So T[i, j, k] = 1 if (i,j) and (j,k) are the inputs to the output (k,i).
    # Yes.
    
    # Create the full tensor T of shape (n, m, p)
    # T[i, j, k] = 1
    # Wait, C[k, i] depends on A[i, j] and B[j, k].
    # So for a fixed i, j, k, the term A[i, j] B[j, k] contributes to C[k, i].
    # So T[i, j, k] = 1.
    
    T = np.ones((n, m, p))
    
    # We want to find U (n*m, R), V (m*p, R), W (p*n, R)
    # such that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i, j, k]
    
    # Reshape T to match the linear system structure
    # Let's flatten indices: 
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    
    # We will try ranks from 54 up to 62
    best_rank = n * m * p
    best_factors = None
    
    # Schoolbook fallback
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    U_fb, V_fb, W_fb, R_fb = schoolbook()
    
    # Try to find lower rank
    # Use Randomized ALS with multiple restarts
    np.random.seed(42)
    
    for R in range(54, 63):
        if time.time() > deadline:
            break
        
        best_err_for_R = float('inf')
        best_U_V_W_for_R = None
        
        n_restarts = 20 if R <= 55 else 10
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with different strategies
            if restart % 3 == 0:
                U = np.random.randn(n * m, R)
                V = np.random.randn(m * p, R)
                W = np.random.randn(p * n, R)
            elif restart % 3 == 1:
                U = np.random.randn(n * m, R) * 0.5
                V = np.random.randn(m * p, R) * 0.5
                W = np.random.randn(p * n, R) * 0.5
            else:
                U = np.ones((n * m, R)) * 0.1
                V = np.ones((m * p, R)) * 0.1
                W = np.ones((p * n, R)) * 0.1
        
        # Pre-compute structures for fast solving
        # We solve for U given V, W.
        # Equations: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i, j, k]
        # This is a linear system for U if V, W are fixed.
        # Vectorize:
        # Let y be the vectorized T.
        # The matrix A_U maps U to the tensor.
        
        # Instead of building the huge matrix every time, we can use the fact that
        # this is a 3-tensor completion/fitting problem.
        # We can use SVD on mode-1, mode-2, mode-3 unfoldings approximately.
        # But ALS is more direct.
        
        # Let's build the matrix A_U once for each R? No, A_U depends on V, W.
        # A_U has size (n*m*p) x (n*m*R)
        # Rows indexed by (i, j, k). Columns by (i*m+j, r).
        # A_U[(i,j,k), (i*m+j, r)] = V[j*p+k, r] * W[k*n+i, r]
        
        # This matrix is sparse? 
        # For a fixed (i,j,k), only rows (i*m+j, r) are non-zero? No, 
        # The term is U[i*m+j, r]. So for a specific (i,j), U[i*m+j, r] appears in all k.
        # So row (i,j,k) has non-zeros only in columns corresponding to i*m+j.
        # Specifically, columns (i*m+j)*R + r.
        # Value is V[j*p+k, r] * W[k*n+i, r].
        
        # This structure allows efficient solving.
        # We can iterate over i, j.
        # For fixed i, j, we have a system for U[i*m+j, :] of size R.
        # The equations involving U[i*m+j, :] are for all k.
        # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = T[i, j, k]
        # Let M_k[r] = V[j*p+k, r] * W[k*n+i, r].
        # Then sum_r U[i*m+j, r] M_k[r] = T[i, j, k].
        # This is a system M @ U_col = T_col, where M is p x R.
        # Since R > p (54 > 7), this is underdetermined. We pick the minimum norm solution.
        # U_col = M^T (M M^T)^{-1} T_col.
        
        # This is very fast!
        
        converged = False
        max_iters = 80
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build M (p x R)
                    M = np.zeros((p, R))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Target vector T_col (p)
                    T_col = np.ones(p)
                    try:
                        MT = M.T
                        MMt = M @ MT
                        # Add small regularization for stability
                        MMt += 1e-9 * np.eye(R)
                        inv = np.linalg.inv(MMt)
                        U_col = MT @ (inv @ np.ones(p))
                        U[i * m + j, :] = U_col
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # Build M (n x R)
                    M = np.zeros((n, R))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    T_col = np.ones(n)
                    try:
                        MT = M.T
                        MMt = M @ MT
                        MMt += 1e-9 * np.eye(R)
                        inv = np.linalg.inv(MMt)
                        V_col = MT @ (inv @ np.ones(n))
                        V[j * p + k, :] = V_col
                    except:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    # Build M (m x R)
                    M = np.zeros((m, R))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    T_col = np.ones(m)
                    try:
                        MT = M.T
                        MMt = M @ MT
                        MMt += 1e-9 * np.eye(R)
                        inv = np.linalg.inv(MMt)
                        W_col = MT @ (inv @ np.ones(m))
                        W[k * n + i, :] = W_col
                    except:
                        pass
                        
            # Calculate Error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err += (val - 1.0) ** 2
                        
            if err < 1e-8:
                converged = True
                break
                
            if err > prev_err * 1.01:  # Allow small increases
                break
            prev_err = err
        
        if converged:
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Verify exactness of snapped version
            snap_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U_snap[i * m + j, r] * V_snap[j * p + k, r] * W_snap[k * n + i, r]
                        snap_err += (val - 1.0) ** 2
            
            if snap_err < 1e-6:
                if best_err_for_R > snap_err:
                    best_err_for_R = snap_err
                    best_U_V_W_for_R = (U_snap.copy(), V_snap.copy(), W_snap.copy())
                if R == 54:
                    best_rank = R
                    best_factors = best_U_V_W_for_R
                    break
        else:
            # Even if not fully converged, check if close enough
            if err < 1e-4:
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                snap_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U_snap[i * m + j, r] * V_snap[j * p + k, r] * W_snap[k * n + i, r]
                            snap_err += (val - 1.0) ** 2
                
                if snap_err < 1e-6:
                    if best_err_for_R > snap_err:
                        best_err_for_R = snap_err
                        best_U_V_W_for_R = (U_snap.copy(), V_snap.copy(), W_snap.copy())
                    if R == 54:
                        best_rank = R
                        best_factors = best_U_V_W_for_R
                        break
    
    if best_factors is None and best_U_V_W_for_R is not None:
        best_rank = R
        best_factors = best_U_V_W_for_R

    if best_factors:
        U, V, W = best_factors
        # Convert to lists of lists as required
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
