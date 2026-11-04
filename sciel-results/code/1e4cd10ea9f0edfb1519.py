import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the all-ones tensor."""
    # Vectorized verification: compute all tensor entries at once
    # Tensor[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape to exploit structure
    U_r = U.reshape(n, m, -1)  # n x m x r
    V_r = V.reshape(m, p, -1)  # m x p x r
    W_r = W.reshape(p, n, -1)  # p x n x r
    
    # For each (i,j,k), compute dot product
    # Use einsum for efficiency
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.all(np.abs(T - 1.0) <= tol)

def polish_to_half_integers(U, V, W, n, m, p, max_iters=200):
    """
    High-precision polish pass: given a near-exact real decomposition,
    try to find an exact half-integer decomposition by local search.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    
    # First, round to nearest half-integer and check
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    # Local search: try flipping individual half-integer values
    for _ in range(max_iters):
        improved = False
        
        # Try modifying U
        for idx in np.random.choice(n * m * r, size=min(100, n * m * r), replace=False):
            row = idx // r
            col = idx % r
            old_val = U_rounded[row, col]
            # Try neighboring half-integers
            for delta in [-1.0, -0.5, 0.5, 1.0]:
                new_val = old_val + delta
                if new_val != old_val:
                    U_rounded[row, col] = new_val
                    if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                        improved = True
                        break
                    U_rounded[row, col] = old_val
            if improved:
                break
        
        if improved:
            continue
        
        # Try modifying V
        for idx in np.random.choice(m * p * r, size=min(100, m * p * r), replace=False):
            row = idx // r
            col = idx % r
            old_val = V_rounded[row, col]
            for delta in [-1.0, -0.5, 0.5, 1.0]:
                new_val = old_val + delta
                if new_val != old_val:
                    V_rounded[row, col] = new_val
                    if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                        improved = True
                        break
                    V_rounded[row, col] = old_val
            if improved:
                break
        
        if improved:
            continue
        
        # Try modifying W
        for idx in np.random.choice(p * n * r, size=min(100, p * n * r), replace=False):
            row = idx // r
            col = idx % r
            old_val = W_rounded[row, col]
            for delta in [-1.0, -0.5, 0.5, 1.0]:
                new_val = old_val + delta
                if new_val != old_val:
                    W_rounded[row, col] = new_val
                    if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                        improved = True
                        break
                    W_rounded[row, col] = old_val
            if improved:
                break
        
        if not improved:
            break
    
    if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 140  # Safety margin (evaluator kills at 150s)
    
    best_rank = n * m * p
    best_decomp = None
    
    # Pre-compute schoolbook as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try to find lower rank using vectorized ALS
    # Start from higher ranks and work down
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But we can likely do better. Start from 30 and go down.
    ranks_to_try = list(range(30, best_rank))
    
    for r in ranks_to_try:
        if time.time() - start_time > timeout:
            break
        
        # Try multiple restarts for this rank
        # Increase restarts for lower ranks where search is harder
        max_restarts = 50 if r <= 35 else 30
        found = False
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Initialize with different strategies
            strategy = restart % 4
            if strategy == 0:
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
            elif strategy == 1:
                U = np.random.randn(n * m, r) * 1.0
                V = np.random.randn(m * p, r) * 1.0
                W = np.random.randn(p * n, r) * 1.0
            elif strategy == 2:
                # Start from half-integer values
                U = np.random.randint(0, 4, (n * m, r)) * 0.5
                V = np.random.randint(0, 4, (m * p, r)) * 0.5
                W = np.random.randint(0, 4, (p * n, r)) * 0.5
            else:
                # Sparse initialization
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
                U[np.random.rand(n * m, r) < 0.5] = 0
                V[np.random.rand(m * p, r) < 0.5] = 0
                W[np.random.rand(p * n, r) < 0.5] = 0
            
            # Vectorized ALS iterations
            converged = False
            best_err = np.inf
            for it in range(150):
                if time.time() - start_time > timeout:
                    break
                
                # Update U: for each (i,j), solve min ||A u - 1||^2
                # where A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                for i in range(n):
                    for j in range(m):
                        # Build A vectorized
                        V_sub = V[j * p:(j + 1) * p, :]  # p x r
                        # W_sub: rows k*n+i for k=0..p-1
                        W_sub = W.reshape(p, n, r)[:, i, :]  # p x r
                        A = V_sub * W_sub  # p x r
                        # Solve A u = 1
                        u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                
                # Update V: for each (j,k), solve min ||A v - 1||^2
                # where A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                for j in range(m):
                    for k in range(p):
                        # U_sub: rows i*m+j for i=0..n-1
                        U_sub = U.reshape(n, m, r)[:, j, :]  # n x r
                        W_sub = W[k * n:(k + 1) * n, :]  # n x r
                        A = U_sub * W_sub  # n x r
                        v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                
                # Update W: for each (k,i), solve min ||A w - 1||^2
                # where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                for k in range(p):
                    for i in range(n):
                        # U_sub: rows i*m+j for j=0..m-1
                        U_sub = U.reshape(n, m, r)[i, :, :]  # m x r
                        # V_sub: rows j*p+k for j=0..m-1
                        V_sub = V.reshape(m, p, r)[:, k, :]  # m x r
                        A = U_sub * V_sub  # m x r
                        w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                
                # Compute error vectorized using einsum
                U_r = U.reshape(n, m, r)
                V_r = V.reshape(m, p, r)
                W_r = W.reshape(p, n, r)
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                err = np.sum((T - 1.0) ** 2)
                
                if err < best_err:
                    best_err = err
                
                if err < 1e-8:
                    converged = True
                    break
            
            if converged or best_err < 1e-6:
                # Try to polish to exact half-integer solution
                result = polish_to_half_integers(U, V, W, n, m, p)
                if result is not None:
                    U_rounded, V_rounded, W_rounded = result
                    if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                        best_rank = r
                        best_decomp = (U_rounded, V_rounded, W_rounded)
                        found = True
                        break
        
        # If we found a decomposition at this rank, continue to try lower ranks
        # (we don't break the outer loop)
    
    # Return the best found
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
