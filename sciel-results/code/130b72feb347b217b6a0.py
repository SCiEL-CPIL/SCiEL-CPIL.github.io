"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The tensor T has T[(i,k),(i,j),(j,k)] = 1
    # We want rank-r decomposition
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank(r, max_iters=200):
        """Try to find a rank-r decomposition using iterative approach."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Use alternating least squares style approach
        # Fix V and W, solve for U (linear)
        # Fix U and W, solve for V (linear)
        # Fix U and V, solve for W (linear)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
            # This is: U[i*m+j, :] @ diag(V[j*p+k,:]*W[k*n+i,:]) = 1 for each k
            # Actually: for fixed (i,j), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k in range(p)
            # Let M_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product)
            # Then U[i*m+j,:] @ M_k = 1 for all k
            # So we have a system: for each k, sum_c U[i*m+j,c] * M_k[c] = 1
            
            for i in range(n):
                for j in range(m):
                    # Build system: p equations, r unknowns
                    A_mat = np.zeros((p, r))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b_vec = np.ones(p)
                    # Solve least squares
                    if r >= p:
                        U[i*m+j, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    else:
                        # Underdetermined, use minimum norm
                        U[i*m+j, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update V: for each (j,k), V[j*p+k, :] should satisfy
            # sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, r))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b_vec = np.ones(n)
                    V[j*p+k, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update W: for each (k,i), W[k*n+i, :] should satisfy
            # sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, r))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b_vec = np.ones(m)
                    W[k*n+i, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Check convergence
            if it % 10 == 0:
                # Compute residual
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(s - 1.0))
                if max_err < 1e-8:
                    return (U.tolist(), V.tolist(), W.tolist())
        
        # Final check
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(s - 1.0))
        if max_err < 1e-8:
            return (U.tolist(), V.tolist(), W.tolist())
        return None
    
    # Try ranks from low to high
    for r in range(18, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + attempt * 1000 + r)
            result = try_rank(r)
            if result is not None:
                U, V, W = result
                # Verify
                if verify(U, V, W, r):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = r
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
        else:
            # If we can't find rank r, try r+1 might work
            # But for now, if we found a better rank, we can stop
            if best_rank < n * m * p:
                break
    
    return best_result
# EVOLVE-BLOCK-END
