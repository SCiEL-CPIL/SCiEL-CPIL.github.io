import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Build rank-44 solution
    U44, V44, W44 = build_rank44(n, m, p)
    
    # Try to reduce to rank 43 by removing one column and finding a completion
    # Try all 44 columns, prioritizing structural diversity
    # Strassen blocks: cols 0-27 (4 blocks of 7), Trivial blocks: cols 28-43 (4 blocks of 4)
    candidate_cols = list(range(44))
    
    for col in candidate_cols:
        if time.time() > deadline - 10:
            break
        result = try_reduce(U44, V44, W44, col, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank-43 not found, return rank-44
    return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}

def build_rank44(n, m, p):
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    def add_122(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 2, 0)
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 2, 2)
    add_122(2, 0, 0)
    add_122(2, 2, 0)
    add_122(2, 0, 2)
    add_122(2, 2, 2)

    return U, V, W

def check_fast(U, V, W, n, m, p, tol=1e-6):
    """Vectorized check of tensor decomposition."""
    R = U.shape[1]
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # Vectorize: create arrays and use einsum
    # U shape: (n*m, R)
    # V shape: (m*p, R)  
    # W shape: (p*n, R)
    
    # Reshape to 3D
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ik' but we need all i,j,k
    # Actually: result[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # einsum: 'ijr,jkr,kir->ijk'
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    
    return np.all(np.abs(result - 1.0) < tol)

def try_reduce(U, V, W, remove_col, n, m, p, deadline):
    """Try to reduce rank by removing one column and finding a completion."""
    R_new = U.shape[1] - 1  # 43
    
    U43 = np.delete(U, remove_col, axis=1).copy()
    V43 = np.delete(V, remove_col, axis=1).copy()
    W43 = np.delete(W, remove_col, axis=1).copy()
    
    # Precompute index slices for fast access
    # W43 has shape (p*n, R_new). For fixed i, rows k*n+i for k=0..p-1 are W43[i::n, :]
    # V43 has shape (m*p, R_new). For fixed j, rows j*p..j*p+p-1 are V43[j*p:(j+1)*p, :]
    # U43 has shape (n*m, R_new). For fixed (i,j), row i*m+j
    
    ones_p = np.ones(p)
    ones_n = np.ones(n)
    ones_m = np.ones(m)
    
    # ALS iterations to find a valid rank-43 decomposition
    for it in range(60):
        if time.time() > deadline - 5:
            break
        
        # Update U: for each (i,j), solve M^T * u = ones(p)
        # M[k, r] = V43[j*p+k, r] * W43[k*n+i, r]
        # Vectorized: V_slice = V43[j*p:(j+1)*p, :] shape (p, R)
        #             W_slice = W43[i::n, :] shape (p, R)
        #             M = V_slice * W_slice shape (p, R)
        for i in range(n):
            W_slice = W43[i::n, :]  # shape (p, R_new)
            for j in range(m):
                V_slice = V43[j*p:(j+1)*p, :]  # shape (p, R_new)
                M = V_slice * W_slice  # shape (p, R_new)
                try:
                    sol, *_ = np.linalg.lstsq(M.T, ones_p, rcond=None)
                    U43[i * m + j, :] = sol
                except:
                    pass
        
        # Update V: for each (j,k), solve M^T * v = ones(n)
        # M[i, r] = U43[i*m+j, r] * W43[k*n+i, r]
        for j in range(m):
            for k in range(p):
                W_row = W43[k*n:(k+1)*n, :]  # shape (n, R_new)
                U_slice = U43[j::m, :]  # shape (n, R_new) - rows i*m+j for i=0..n-1
                M = U_slice * W_row  # shape (n, R_new)
                try:
                    sol, *_ = np.linalg.lstsq(M.T, ones_n, rcond=None)
                    V43[j * p + k, :] = sol
                except:
                    pass
        
        # Update W: for each (k,i), solve M^T * w = ones(m)
        # M[j, r] = U43[i*m+j, r] * V43[j*p+k, r]
        for k in range(p):
            for i in range(n):
                V_row = V43[k::p, :]  # shape (m, R_new) - rows j*p+k for j=0..m-1
                U_row = U43[i*m:(i+1)*m, :]  # shape (m, R_new)
                M = U_row * V_row  # shape (m, R_new)
                try:
                    sol, *_ = np.linalg.lstsq(M.T, ones_m, rcond=None)
                    W43[k * n + i, :] = sol
                except:
                    pass
        
        # Check if we have a valid solution
        if check_fast(U43, V43, W43, n, m, p, tol=1e-5):
            # Polish: do a few more iterations with tighter tolerance
            for polish in range(5):
                for i in range(n):
                    W_slice = W43[i::n, :]
                    for j in range(m):
                        V_slice = V43[j*p:(j+1)*p, :]
                        M = V_slice * W_slice
                        try:
                            sol, *_ = np.linalg.lstsq(M.T, ones_p, rcond=None)
                            U43[i * m + j, :] = sol
                        except:
                            pass
                for j in range(m):
                    for k in range(p):
                        W_row = W43[k*n:(k+1)*n, :]
                        U_slice = U43[j::m, :]
                        M = U_slice * W_row
                        try:
                            sol, *_ = np.linalg.lstsq(M.T, ones_n, rcond=None)
                            V43[j * p + k, :] = sol
                        except:
                            pass
                for k in range(p):
                    for i in range(n):
                        V_row = V43[k::p, :]
                        U_row = U43[i*m:(i+1)*m, :]
                        M = U_row * V_row
                        try:
                            sol, *_ = np.linalg.lstsq(M.T, ones_m, rcond=None)
                            W43[k * n + i, :] = sol
                        except:
                            pass
            
            # Round to half-integers for cleaner output
            U43_r = np.round(U43 * 2) / 2
            V43_r = np.round(V43 * 2) / 2
            W43_r = np.round(W43 * 2) / 2
            if check_fast(U43_r, V43_r, W43_r, n, m, p, tol=1e-6):
                return U43_r, V43_r, W43_r
            # Also try rounding to integers
            U43_i = np.round(U43)
            V43_i = np.round(V43)
            W43_i = np.round(W43)
            if check_fast(U43_i, V43_i, W43_i, n, m, p, tol=1e-6):
                return U43_i, V43_i, W43_i
            # Return unrounded if close enough
            if check_fast(U43, V43, W43, n, m, p, tol=1e-8):
                return U43, V43, W43
    
    return None
