import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W form a valid decomposition with integer/half-integer entries."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check integer/half-integer constraint first (cheap)
    # Entries must be in 0.5 * Z
    U_h = U_arr * 2
    V_h = V_arr * 2
    W_h = W_arr * 2
    
    if not np.allclose(U_h, np.round(U_h), atol=1e-5):
        return False
    if not np.allclose(V_h, np.round(V_h), atol=1e-5):
        return False
    if not np.allclose(W_h, np.round(W_h), atol=1e-5):
        return False

    # Check exact decomposition
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Indices: a = i*m+j, b = j*p+k, c = k*n+i
    # T is 1 if indices are consistent, 0 otherwise.
    
    # Efficient check using einsum
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Build target tensor for comparison
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
        
    return True

def _als_decompose(rank, n, m, p, max_iters=1000, seed=42, tol=1e-10):
    """
    Alternating Least Squares for tensor decomposition.
    Returns (U, V, W) or None.
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    # Normalize initial vectors
    for r in range(rank):
        norm_u = np.linalg.norm(U[:, r])
        norm_v = np.linalg.norm(V[:, r])
        norm_w = np.linalg.norm(W[:, r])
        if norm_u > 1e-8: U[:, r] /= norm_u
        if norm_v > 1e-8: V[:, r] /= norm_v
        if norm_w > 1e-8: W[:, r] /= norm_w

    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            return None
            
        # Update U
        # For each i, j, we solve for U[i*m+j, :]
        # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
        # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
        # Then A @ U[i*m+j, :]^T = ones(p)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ u = 1
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[i*m+j, :] = u
                
        # Update V
        # For each j, k, solve for V[j*p+k, :]
        # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
        # Let B[i, r] = U[i*m+j, r] * W[k*n+i, r]
        # Then B @ v = ones(n)
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                V[j*p+k, :] = v
                
        # Update W
        # For each k, i, solve for W[k*n+i, :]
        # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
        # Let C[j, r] = U[i*m+j, r] * V[j*p+k, r]
        # Then C @ w = ones(m)
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                w, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                W[k*n+i, :] = w
                
        # Normalize to prevent overflow/underflow
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-8: U[:, r] /= norm_u
            if norm_v > 1e-8: V[:, r] /= norm_v
            if norm_w > 1e-8: W[:, r] /= norm_w

        # Check convergence
        if it % 10 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1.0
            err = np.max(np.abs(T_recon - T_target))
            if err < tol:
                # Try rounding
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                
    # Final check and rounding
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    err = np.max(np.abs(T_recon - T_target))
    
    if err < 1e-5:
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
            
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    n, m, p = 2, 3, 4
    schoolbook_rank = n * m * p
    
    # Try ranks lower than schoolbook
    # Lower bound is max(n*m, m*p, n*p) = 12
    # We try 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12
    
    ranks_to_try = list(range(schoolbook_rank - 1, 11, -1))
    
    # Limit attempts per rank to fit within time
    # We have ~140s. 
    # ALS is relatively fast for small dimensions.
    # Let's allocate time based on rank. Lower ranks are harder, might need more restarts.
    
    for rank in ranks_to_try:
        if time.time() > _DEADLINE - 10:
            break
            
        # Number of seeds to try
        # If rank is close to schoolbook, it's easier.
        # If rank is low, it's harder.
        num_seeds = 50 if rank > 20 else 20
        
        for seed in range(num_seeds):
            if time.time() > _DEADLINE - 5:
                break
                
            res = _als_decompose(rank, n, m, p, max_iters=200, seed=seed)
            if res is not None:
                U, V, W = res
                # Double check verification
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
