import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k].
    Convention: W index is k*n + i, so T[i,j,k] with i in [0,n), j in [0,m), k in [0,p).
    U index: i*m + j, V index: j*p + k, W index: k*n + i.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _verify(U, V, W):
    """Verify exact decomposition. U: (nm, r), V: (mp, r), W: (pn, r).
    Check that sum_r U[:,r] * V[:,r] * W[:,r] == T.
    """
    n, m, p = DIMENSIONS
    T = _build_tensor()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    # Reconstruct: T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Efficient: for each r, outer product
    T_recon = np.zeros((n * m, m * p, p * n))
    for r_idx in range(r):
        T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
    return np.allclose(T_recon, T, atol=1e-10)

def _round_to_halfint(U, V, W):
    """Round entries to nearest half-integer."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r.tolist(), V_r.tolist(), W_r.tolist()

def _als_decompose(T, rank, max_iter=200, tol=1e-12, seed=42):
    """ALS decomposition of tensor T into rank factors.
    T: (nm, mp, pn)
    Returns U: (nm, rank), V: (mp, rank), W: (pn, rank)
    """
    np.random.seed(seed)
    nm, mp, pn = T.shape
    
    # Initialize
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    for it in range(max_iter):
        # Update U: for each r, U[:,r] = (T * (V[:,r] ⊗ W[:,r])) / (||V[:,r]||^2 * ||W[:,r]||^2)
        # T flattened as (nm, mp*pn), factor = V[:,r] * W[:,r] reshaped
        U_new = np.zeros_like(U)
        for r in range(rank):
            factor = (V[:, r] * W[:, r]).reshape(mp, pn)
            # T reshaped: (nm, mp, pn) -> contract with factor over mp,pn
            U_new[:, r] = np.einsum('abc,bc->a', T, factor)
            denom = np.sum(V[:, r] ** 2) * np.sum(W[:, r] ** 2)
            if denom > 1e-15:
                U_new[:, r] /= denom
        
        # Update V
        V_new = np.zeros_like(V)
        for r in range(rank):
            factor = (U[:, r] * W[:, r]).reshape(nm, pn)
            V_new[:, r] = np.einsum('abc,ac->b', T, factor)
            denom = np.sum(U[:, r] ** 2) * np.sum(W[:, r] ** 2)
            if denom > 1e-15:
                V_new[:, r] /= denom
        
        # Update W
        W_new = np.zeros_like(W)
        for r in range(rank):
            factor = (U[:, r] * V[:, r]).reshape(nm, mp)
            W_new[:, r] = np.einsum('abc,ab->c', T, factor)
            denom = np.sum(U[:, r] ** 2) * np.sum(V[:, r] ** 2)
            if denom > 1e-15:
                W_new[:, r] /= denom
        
        # Check convergence
        delta = (np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W))
        U, V, W = U_new, V_new, W_new
        if delta < tol:
            break
    
    return U, V, W

def _try_rank(T, rank, max_iter=100, n_trials=5, deadline=None):
    """Try to find exact rank-r decomposition. Returns (U,V,W) or None."""
    best_result = None
    for trial in range(n_trials):
        if deadline and time.time() > deadline:
            break
        U, V, W = _als_decompose(T, rank, max_iter=max_iter, seed=42 + trial)
        # Round to half-integers
        U_r, V_r, W_r = _round_to_halfint(U, V, W)
        if _verify(U_r, V_r, W_r):
            best_result = (U_r, V_r, W_r)
            return best_result
    return best_result

def _schoolbook():
    """Return the schoolbook rank-80 decomposition."""
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    T = _build_tensor()
    
    # Start with schoolbook as fallback
    fallback = _schoolbook()
    
    # Try ranks from low to high
    # Theoretical lower bound: for 4x4x4x5, rank >= 12 (from known bounds)
    # Known: 4x4x4x4 has rank 15 (Strassen-like), so 4x4x4x5 might be similar
    # Try ranks: 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 40, 50, 60, 80
    
    ranks_to_try = [12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25, 30, 40, 50, 60, 70, 80]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        result = _try_rank(T, rank, max_iter=80, n_trials=3, deadline=deadline)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r, "V": V_r, "W": W_r}
    
    return fallback

