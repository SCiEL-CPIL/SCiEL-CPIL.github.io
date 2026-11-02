"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # Build the tensor T[i, j, k] = 1 if C[i,k] gets contribution from A[i,j]*B[j,k]
    # T is (n, m, p) tensor where T[i,j,k] = 1 for all i,j,k
    # Actually the matrix multiplication tensor: T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
    # The tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # W index: W[k*n + i, r]
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Tensor entries: T[i,j,k] = 1 for all i,j,k
    T = np.ones((n, m, p))
    
    def verify(U, V, W):
        """Check if U,V,W exactly decompose the tensor."""
        rank = U.shape[1]
        # Reconstruct T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        recon = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U[:, r]  # (n*m,)
            v_col = V[:, r]  # (m*p,)
            w_col = W[:, r]  # (p*n,)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        return np.allclose(recon, T, atol=1e-9)
    
    def verify_fast(U, V, W):
        """Faster verification using matrix operations."""
        rank = U.shape[1]
        recon = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # u_col is (n*m,), reshape to (n, m)
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            # recon[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        return np.allclose(recon, T, atol=1e-9)
    
    def als_decompose(rank, init_U=None, init_V=None, init_W=None, max_iter=200, tol=1e-10):
        """ALS for tensor decomposition T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]"""
        if init_U is not None:
            U = init_U.copy().astype(float)
        else:
            U = np.random.randn(n*m, rank) * 0.1
        if init_V is not None:
            V = init_V.copy().astype(float)
        else:
            V = np.random.randn(m*p, rank) * 0.1
        if init_W is not None:
            W = init_W.copy().astype(float)
        else:
            W = np.random.randn(p*n, rank) * 0.1
        
        prev_error = float('inf')
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), U[i*m+j, :] = (V[j*p+k,:] * W[k*n+i,:])_k dot T[i,j,:]
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # For fixed (i,j), T[i,j,:] = U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:])_k
            # This is a linear system: T[i,j,:] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
            # Let M[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
            # Then T[i,j,:] = U[i*m+j,:] @ M[i,j,:,:] (where M is (p, rank))
            for i in range(n):
                for j in range(m):
                    # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    M = np.outer(V[j*p:(j+1)*p, :].T, W[:, :].T)  # This is wrong
                    # Let me redo: V[j*p+k, r] for k in range(p), r in range(rank) -> shape (p, rank)
                    # W[k*n+i, r] for k in range(p), r in range(rank) -> shape (p, rank)
                    V_sub = V[j*p:(j+1)*p, :]  # (p, rank)
                    W_sub = W[0::n, :]  # This is wrong, W[k*n+i] for k in range(p)
                    W_sub = W[np.array([k*n+i for k in range(p)]), :]  # (p, rank)
                    M = V_sub * W_sub  # (p, rank)
                    # T[i,j,:] = U[i*m+j,:] @ M.T -> T[i,j,k] = sum_r U[i*m+j,r] * M[k,r]
                    # Solve: U[i*m+j,:] = T[i,j,:] @ M @ (M.T @ M)^{-1}
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, T[i,j,:], rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), V[j*p+k, :] = (U[i*m+j,:] * W[k*n+i,:])_i dot T[:,j,k]
            for j in range(m):
                for k in range(p):
                    # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    U_sub = U[np.array([i*m+j for i in range(n)]), :]  # (n, rank)
                    W_sub = W[np.array([k*n+i for i in range(n)]), :]  # (n, rank)
                    M = U_sub * W_sub  # (n, rank)
                    # T[:,j,k] = V[j*p+k,:] @ M.T
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M.T, T[:,j,k], rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), W[k*n+i, :] = (U[i*m+j,:] * V[j*p+k,:])_j dot T[i,:,k]
            for k in range(p):
                for i in range(n):
                    # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    U_sub = U[np.array([i*m+j for j in range(m)]), :]  # (m, rank)
                    V_sub = V[np.array([j*p+k for j in range(m)]), :]  # (m, rank)
                    M = U_sub * V_sub  # (m, rank)
                    # T[i,:,k] = W[k*n+i,:] @ M.T
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, T[i,:,k], rcond=None)[0]
                    except:
                        pass
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        error += (val - T[i,j,k])**2
            error = np.sqrt(error)
            
            if abs(prev_error - error) < tol:
                break
            prev_error = error
        
        return U, V, W, error
    
    def round_to_half(U, V, W):
        """Round entries to nearest half-integer."""
        U_r = np.round(2*U) / 2.0
        V_r = np.round(2*V) / 2.0
        W_r = np.round(2*W) / 2.0
        return U_r, V_r, W_r
    
    def to_list(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to find a lower rank decomposition
    best_result = None
    best_rank = n * m * p  # 80
    
    # Get schoolbook as initial
    U_sb, V_sb, W_sb = schoolbook()
    
    # Try various ranks with ALS
    # Start from ranks below 80
    for target_rank in [72, 70, 68, 64, 60, 56, 52, 49, 45, 40, 36, 32, 28, 24, 20]:
        if time.time() > deadline - 10:
            break
        
        # Multiple random initializations
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + target_rank * 100 + trial)
            U, V, W, error = als_decompose(
                target_rank,
                init_U=None,
                init_V=None,
                init_W=None,
                max_iter=100
            )
            
            # Round to half-integers
            U_r, V_r, W_r = round_to_half(U, V, W)
            
            # Verify
            if verify_fast(U_r, V_r, W_r):
                best_result = to_list(U_r, V_r, W_r)
                best_rank = target_rank
                break
        
        if best_result is not None:
            break
    
    # If no lower rank found, try to reduce schoolbook rank
    # by zeroing out some columns and re-fitting
    if best_result is None:
        # Try trimming schoolbook
        for trim in range(1, 20):
            if time.time() > deadline - 5:
                break
            target_rank = 80 - trim
            if target_rank <= 0:
                break
            
            # Initialize with first target_rank columns of schoolbook
            U_init = U_sb[:, :target_rank].copy()
            V_init = V_sb[:, :target_rank].copy()
            W_init = W_sb[:, :target_rank].copy()
            
            U, V, W, error = als_decompose(
                target_rank,
                init_U=U_init,
                init_V=V_init,
                init_W=W_init,
                max_iter=50
            )
            
            U_r, V_r, W_r = round_to_half(U, V, W)
            
            if verify_fast(U_r, V_r, W_r):
                best_result = to_list(U_r, V_r, W_r)
                best_rank = target_rank
                break
    
    # Final fallback: schoolbook
    if best_result is None:
        best_result = to_list(U_sb, V_sb, W_sb)
    
    return best_result
# EVOLVE-BLOCK-END
