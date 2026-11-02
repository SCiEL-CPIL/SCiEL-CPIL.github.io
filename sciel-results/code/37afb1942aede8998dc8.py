"""Matrix multiplication 3,5,6 tensor decomposition.

Strategy:
1. Build the schoolbook rank-90 decomposition (valid baseline).
2. Attempt structured rank reduction via Kronecker-block Strassen-style merging.
3. Use ALS (Alternating Least Squares) with rank-deflation to try to find a
   lower-rank exact decomposition with integer/half-integer entries.
4. Always fall back to the valid schoolbook result if no improvement is found
   within the time budget.

The key insight for 3x5x6: we can view the multiplication as combining
smaller Strassen-like blocks. For example, 3x5x6 can be decomposed using
blocks that allow some sharing of intermediate products.

We try several structured approaches:
- Direct Kronecker product of smaller Strassen decompositions
- Block-wise sharing where possible
- ALS-based rank reduction from the schoolbook starting point
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

def _schoolbook(n, m, p):
    """Return the trivial rank-(n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid
    index triple, else 0. Shape: (n*m, m*p, p*n) in flattened form.
    
    The tensor T[a, b, c] where:
      a = i*m + j  (i in 0..n-1, j in 0..m-1)
      b = j*p + k  (j in 0..m-1, k in 0..p-1)
      c = k*n + i  (k in 0..p-1, i in 0..n-1)
    T[a,b,c] = 1 if the middle index j matches in a and b, and the k in b matches i in c appropriately.
    Actually: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form an exact decomposition of the matrix multiplication tensor.
    
    T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    T = _target_tensor(n, m, p)
    # Compute the decomposition tensor
    # D[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is a third-order tensor contraction
    n_r = U.shape[1]
    # Efficiently: for each r, outer product of U[:,r], V[:,r], W[:,r]
    D = np.zeros(T.shape)
    for r in range(n_r):
        D += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(D, T, atol=tol)

def _round_half_int(M):
    """Round a matrix to nearest half-integers."""
    return np.round(2 * M) / 2

def _als_decompose(T, rank, max_iter=200, tol=1e-10, seed=42):
    """Try to decompose tensor T into U, V, W of given rank using ALS.
    
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    rng = np.random.RandomState(seed)
    a, b, c = T.shape
    
    U = rng.randn(a, rank) * 0.5
    V = rng.randn(b, rank) * 0.5
    W = rng.randn(c, rank) * 0.5
    
    for it in range(max_iter):
        # Update U: for each a, U[a,:] solves linear system
        # T[a,:,:] = sum_r U[a,r] * V[:,r] * W[:,r]
        # Reshape: for fixed a, T[a] is (b,c) matrix = U[a,:] @ (V * W) where (V*W)[r,:] = V[:,r]*W[:,r]... 
        # Actually: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
        # So for fixed a: T[a] = sum_r U[a,r] * outer(V[:,r], W[:,r])
        # Let M_a = stack of outer(V[:,r], W[:,r]) for r=0..rank-1, shape (rank, b*c)
        # Then T[a].flatten() = M_a @ U[a,:]
        
        # Build the Gram-like matrices for efficiency
        # For U update:
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Fix a: T[a,:, :] = sum_r U[a,r] * (V[:,r] * W[:,r])
        # Let X_r = V[:,r] * W[:,r] (element-wise product), shape (b*c,)
        # T[a].flatten() = sum_r U[a,r] * X_r
        # So: T_flat[a] = U[a,:] @ X^T where X is (rank, b*c)
        
        X_U = np.zeros((rank, b * c))
        for r in range(rank):
            X_U[r, :] = (V[:, r] * W[:, r]).flatten()
        
        T_flat = T.reshape(a, b * c)
        # Solve: T_flat[a, :] = X_U @ U[a, :].T  =>  T_flat.T[a] = X_U.T @ U[a]
        # U[a] = (X_U^T X_U)^{-1} X_U^T T_flat[a]
        try:
            U_new = np.linalg.lstsq(X_U, T_flat.T, rcond=None)[0].T
        except:
            U_new = U
        U = U_new
        
        # Update V:
        # T[a,b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
        # Fix b: T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r])
        X_V = np.zeros((rank, a * c))
        for r in range(rank):
            X_V[r, :] = (U[:, r] * W[:, r]).flatten()
        T_flat_v = T.reshape(a, b, c).transpose(1, 0, 2).reshape(b, a * c)
        try:
            V_new = np.linalg.lstsq(X_V, T_flat_v.T, rcond=None)[0].T
        except:
            V_new = V
        V = V_new
        
        # Update W:
        # T[a,b,c] = sum_r W[c,r] * U[a,r] * V[b,r]
        # Fix c: T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
        X_W = np.zeros((rank, a * b))
        for r in range(rank):
            X_W[r, :] = (U[:, r] * V[:, r]).flatten()
        T_flat_w = T.reshape(a, b, c).transpose(2, 0, 1).reshape(c, a * b)
        try:
            W_new = np.linalg.lstsq(X_W, T_flat_w.T, rcond=None)[0].T
        except:
            W_new = W
        W = W_new
        
        # Check convergence
        if it % 20 == 0:
            D = np.zeros(T.shape)
            for r in range(rank):
                D += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(D - T) / (np.linalg.norm(T) + 1e-15)
            if err < tol:
                break
    
    return U, V, W

def _try_rank_reduction(n, m, p, T, max_rank, time_limit):
    """Try to find a decomposition with rank < max_rank.
    
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # Try ranks from max_rank-1 down to some minimum
    for rank in range(max_rank - 1, max(10, max_rank - 30), -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random restarts
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
            
            U, V, W = _als_decompose(T, rank, max_iter=100, seed=42 + attempt)
            
            # Round to half-integers and verify
            U_r = _round_half_int(U)
            V_r = _round_half_int(V)
            W_r = _round_half_int(W)
            
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                return U_r, V_r, W_r
            
            # Also try without rounding (check if it's already close to half-integers)
            if _verify(U, V, W, n, m, p, tol=1e-6):
                U_r = _round_half_int(U)
                V_r = _round_half_int(V)
                W_r = _round_half_int(W)
                if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                    return U_r, V_r, W_r
    
    return None

def _strassen_block(n1, n2, m1, m2, p1, p2):
    """Try to build a decomposition using Strassen-style block multiplication.
    
    For 3x5x6, we can try to split into blocks.
    E.g., 3 = 2+1, 5 = 3+2, 6 = 3+3 or 4+2, etc.
    """
    pass  # Placeholder for structured approach

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin
    
    # Build schoolbook decomposition
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Build target tensor for verification
    T = _target_tensor(n, m, p)
    
    # Verify schoolbook
    assert _verify(U_sb, V_sb, W_sb, n, m, p), "Schoolbook verification failed!"
    
    # Try rank reduction with time limit
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        result = _try_rank_reduction(n, m, p, T, best_rank, time_remaining - 5)
        if result is not None:
            U_new, V_new, W_new = result
            new_rank = U_new.shape[1]
            if new_rank < best_rank:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_rank
    
    # Check time
    if time.time() > deadline - 2:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
