"""Matrix multiplication 3x3x7 tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We seek U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def verify(U, V, W, r):
        """Verify the decomposition with high precision."""
        # Convert to long double for verification
        U_ld = U.astype(np.longdouble)
        V_ld = V.astype(np.longdouble)
        W_ld = W.astype(np.longdouble)
        
        # Check all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U_ld[i * m + j, r_idx] * V_ld[j * p + k, r_idx] * W_ld[k * n + i, r_idx]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integer(M):
        """Snap entries to nearest 0.5."""
        return np.round(M * 2) / 2

    def try_rank(r, max_iterations=200):
        """Try to find a rank-r decomposition using ALS with high precision."""
        rng = np.random.RandomState(42 + r)
        
        # Initialize with long double
        U = rng.randn(n * m, r).astype(np.longdouble)
        V = rng.randn(m * p, r).astype(np.longdouble)
        W = rng.randn(p * n, r).astype(np.longdouble)
        
        # Scale
        U /= np.sqrt(r)
        V /= np.sqrt(r)
        W /= np.sqrt(r)
        
        best_error = np.inf
        best_U, best_V, best_W = None, None, None
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Update U
            U_new = np.zeros((n * m, r), dtype=np.longdouble)
            for i in range(n):
                for j in range(m):
                    # M[k, r_idx] = V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    M = np.zeros((p, r), dtype=np.longdouble)
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve U[i*m+j, :] @ M = ones(p)
                    # M.T @ U_row = ones
                    try:
                        U[i * m + j, :], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p, dtype=np.longdouble), rcond=None)
                    except:
                        pass
            
            # Update V
            V_new = np.zeros((m * p, r), dtype=np.longdouble)
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r), dtype=np.longdouble)
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(n, dtype=np.longdouble), rcond=None)
                    except:
                        pass
            
            # Update W
            W_new = np.zeros((p * n, r), dtype=np.longdouble)
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r), dtype=np.longdouble)
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(m, dtype=np.longdouble), rcond=None)
                    except:
                        pass
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        error += (val - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-12:
                break
            
            # Damping
            U = 0.5 * U + 0.5 * U_new
            V = 0.5 * V + 0.5 * V_new
            W = 0.5 * W + 0.5 * W_new
        
        # Final snap and verify
        if best_U is not None:
            U_snap = snap_to_half_integer(best_U)
            V_snap = snap_to_half_integer(best_V)
            W_snap = snap_to_half_integer(best_W)
            
            if verify(U_snap, V_snap, W_snap, r):
                return True, U_snap, V_snap, W_snap
            
            # Try without snap, just high precision float
            if verify(best_U, best_V, best_W, r):
                return True, best_U, best_V, best_W
                
        return False, None, None, None

    # Try ranks from low to high
    # Known: 3x3x3 is 23. 3x3x7 is likely around 50-60.
    # We try 59, 58, ..., 63.
    
    best_rank = 63
    best_result = None
    
    for r in [59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 60, 61, 62, 63]:
        if time.time() > deadline - 20:
            break
            
        success, U, V, W = try_rank(r)
        
        if success:
            best_rank = r
            # Convert to lists of floats
            U_list = [[float(U[i, j]) for j in range(r)] for i in range(n * m)]
            V_list = [[float(V[i, j]) for j in range(r)] for i in range(m * p)]
            W_list = [[float(W[i, j]) for j in range(r)] for i in range(p * n)]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    if best_result is None:
        # Fall back to naive
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
