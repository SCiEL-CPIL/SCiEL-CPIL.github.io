"""
Matrix multiplication 4,4,5 rank minimization.
Strategy: Start with Strassen+schoolbook (rank 65), then use ALS refinement
to search for lower-rank decompositions with half-integer entries.
"""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def _strassen_2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 Strassen.
    """
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(0,0): -1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(1,1): 1})
    ]
    return terms

def _build_4x4x4_strassen():
    """
    Constructs 4x4x4 decomposition using two levels of Strassen.
    Rank 49.
    """
    n = 4
    base_terms = _strassen_2x2_terms()
    rank = 49
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    for t1 in range(7):
        U1, V1, W1 = base_terms[t1]
        for t2 in range(7):
            U2, V2, W2 = base_terms[t2]
            idx = t1 * 7 + t2
            
            for (i1, j1), c1 in U1.items():
                for (i2, j2), c2 in U2.items():
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    U[i*n + j, idx] += c1 * c2
            
            for (j1, k1), c1 in V1.items():
                for (j2, k2), c2 in V2.items():
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    V[j*n + k, idx] += c1 * c2
            
            for (k1, i1), c1 in W1.items():
                for (k2, i2), c2 in W2.items():
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    W[k*n + i, idx] += c1 * c2
                    
    return U, V, W

def _build_4x4x5_decomposition():
    """
    Combines 4x4x4 Strassen (rank 49) and 4x4x1 Schoolbook (rank 16).
    Returns U, V, W matrices.
    """
    n, m, p = 4, 4, 5
    
    U444, V444, W444 = _build_4x4x4_strassen()
    rank_444 = 49
    rank_441 = n * m
    total_rank = rank_444 + rank_441
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    U[:, :rank_444] = U444
    V[:m*4, :rank_444] = V444
    W[:4*n, :rank_444] = W444
    
    for i in range(n):
        for j in range(m):
            r = rank_444 + i * m + j
            U[i*m + j, r] = 1.0
            V[j*p + 4, r] = 1.0
            W[4*n + i, r] = 1.0
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=1e-9)

def _round_to_half_integers(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _als_refine(U, V, W, n, m, p, target_rank, max_iters=50, tol=1e-10):
    """
    Try to refine decomposition to target_rank using ALS.
    Returns (U, V, W) if successful, None otherwise.
    """
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return U[:, :target_rank], V[:, :target_rank], W[:, :target_rank]
    
    # Initialize by truncating (keep first target_rank columns)
    U_r = U[:, :target_rank].copy()
    V_r = V[:, :target_rank].copy()
    W_r = W[:, :target_rank].copy()
    
    n, m, p = n, m, p
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    for it in range(max_iters):
        # Update U: for each column i*m+j, solve least squares
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed i,j: T[i,j,:] = U[i*m+j,:] @ (V[j,:,:] * W[:,i,:]).T ... 
                # Actually: T[i,j,k] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
                # Let X_r,k = V[j*p+k,r] * W[k*n+i,r] for r in range(target_rank), k in range(p)
                # Then T[i,j,:] = U[i*m+j,:] @ X_r,k^T? No...
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
                
                X = np.zeros((target_rank, p))
                for k in range(p):
                    X[:, k] = V_r[j*p+k, :] * W_r[k*n+i, :]
                # T[i,j,:] = U[i*m+j,:] @ X.T
                # So U[i*m+j,:] = T[i,j,:] @ X @ (X.T @ X)^{-1}
                target = np.ones(p)  # T[i,j,k] should be 1 for all k
                try:
                    XTX = X.T @ X
                    Xty = X.T @ target
                    U_r[i*m+j, :] = np.linalg.solve(XTX + 1e-12 * np.eye(target_rank), Xty)
                except np.linalg.LinAlgError:
                    pass
        
        # Update V: for each column j*p+k
        for j in range(m):
            for k in range(p):
                X = np.zeros((target_rank, n))
                for i in range(n):
                    X[:, i] = U_r[i*m+j, :] * W_r[k*n+i, :]
                target = np.ones(n)
                try:
                    XTX = X.T @ X
                    Xty = X.T @ target
                    V_r[j*p+k, :] = np.linalg.solve(XTX + 1e-12 * np.eye(target_rank), Xty)
                except np.linalg.LinAlgError:
                    pass
        
        # Update W: for each column k*n+i
        for k in range(p):
            for i in range(n):
                X = np.zeros((target_rank, m))
                for j in range(m):
                    X[:, j] = U_r[i*m+j, :] * V_r[j*p+k, :]
                target = np.ones(m)
                try:
                    XTX = X.T @ X
                    Xty = X.T @ target
                    W_r[k*n+i, :] = np.linalg.solve(XTX + 1e-12 * np.eye(target_rank), Xty)
                except np.linalg.LinAlgError:
                    pass
        
        # Check convergence
        if it > 0:
            # Compute error
            U_r2 = U_r.reshape(n, m, target_rank)
            V_r2 = V_r.reshape(m, p, target_rank)
            W_r2 = W_r.reshape(p, n, target_rank)
            T = np.einsum('ijr,jkr,kir->ijk', U_r2, V_r2, W_r2)
            err = np.max(np.abs(T - 1.0))
            if err < tol:
                break
    
    # Round to half-integers and verify
    U_rounded = _round_to_half_integers(U_r)
    V_rounded = _round_to_half_integers(V_r)
    W_rounded = _round_to_half_integers(W_r)
    
    if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    """
    Returns the factor matrices for the 4x4x5 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_4x4x5_decomposition()
        
        if time.time() > deadline:
            raise TimeoutError()
            
        if _verify(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
            
            # Try to reduce rank using ALS
            for target_rank in range(best_rank - 1, 19, -1):
                if time.time() > deadline - 10:
                    break
                
                result = _als_refine(U, V, W, n, m, p, target_rank, max_iters=30)
                if result is not None:
                    best_U, best_V, best_W = result
                    best_rank = target_rank
                    break  # Found a valid lower-rank decomposition, stop
                
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
    except Exception as e:
        pass
    
    # Schoolbook fallback (Rank 80)
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
