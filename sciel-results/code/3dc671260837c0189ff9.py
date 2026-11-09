import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Build rank-44 solution
    U44, V44, W44 = build_rank44(n, m, p)
    
    # Precompute the tensor structure for fast verification
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # We can vectorize this as a tensor contraction
    
    # Try to reduce to rank 43 by removing one column and finding a completion
    # Strategy: remove a column, then solve for the remaining columns using least squares
    
    # Candidate columns to try removing (prioritize by structural type)
    # Strassen blocks: cols 0-27 (4 blocks of 7)
    # Trivial blocks: cols 28-43 (4 blocks of 4)
    
    # Try removing columns from different structural positions
    # Due to symmetry, try one from each type
    candidate_cols = [
        0,   # Strassen type 0 (M1)
        7,   # Strassen type 0 of block 2
        14,  # Strassen type 0 of block 3
        21,  # Strassen type 0 of block 4
        1,   # Strassen type 1 (M2)
        28,  # Trivial type 0
        29,  # Trivial type 1
        30,  # Trivial type 2
        31,  # Trivial type 3
    ]
    
    for col in candidate_cols:
        if time.time() > deadline - 15:
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
    
    # ALS iterations to find a valid rank-43 decomposition
    for it in range(30):
        if time.time() > deadline - 5:
            break
        
        # Update U: for each (i,j), solve M^T * u = ones
        # where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R_new))
                for k in range(p):
                    M[k, :] = V43[j * p + k, :] * W43[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U43[i * m + j, :] = sol
                except:
                    pass
        
        # Update V: for each (j,k), solve M^T * v = ones
        # where M[i, r] = U[i*m+j, r] * W[k*n+i, r]
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R_new))
                for i in range(n):
                    M[i, :] = U43[i * m + j, :] * W43[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V43[j * p + k, :] = sol
                except:
                    pass
        
        # Update W: for each (k,i), solve M^T * w = ones
        # where M[j, r] = U[i*m+j, r] * V[j*p+k, r]
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R_new))
                for j in range(m):
                    M[j, :] = U43[i * m + j, :] * V43[j * p + k, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W43[k * n + i, :] = sol
                except:
                    pass
        
        # Check if we have a valid solution
        if check_fast(U43, V43, W43, n, m, p, tol=1e-5):
            # Round to half-integers for cleaner output
            U43 = np.round(U43 * 2) / 2
            V43 = np.round(V43 * 2) / 2
            W43 = np.round(W43 * 2) / 2
            if check_fast(U43, V43, W43, n, m, p, tol=1e-6):
                return U43, V43, W43
    
    return None
