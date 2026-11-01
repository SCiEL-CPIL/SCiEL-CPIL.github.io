import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the maximum absolute residual of the tensor decomposition."""
    rank = U.shape[1]
    # The tensor T[i, j, k] should equal sum_r U[i, r] * V[j, r] * W[k, r]
    # where i in [0, n*m), j in [0, m*p), k in [0, p*n)
    # The matrix multiplication tensor has T[i, j, k] = 1 if:
    #   i = a*m + b, j = b*p + c, k = c*n + a for some a, b, c
    # Otherwise T[i, j, k] = 0
    
    # Compute the decomposed tensor
    # T_decomposed[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # This is equivalent to: for each r, outer(U[:, r], outer(V[:, r], W[:, r]))
    
    # Use einsum for efficiency
    # U: (nm, rank), V: (mp, rank), W: (pn, rank)
    # T_decomposed[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    
    # Compute using matrix operations
    # T_decomposed = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # But this might be large. Let's compute it differently.
    # For each pair (i, j), T_decomposed[i, j, :] = (U[i, :] * V[j, :]) @ W.T
    # So we can compute: UV = U @ V.T  (nm x mp)
    # Then T_decomposed[i, j, k] = sum_r (U[i, r] * V[j, r]) * W[k, r]
    # = (U @ V.T)[i, j] ... no, that's not right.
    
    # Actually: T_decomposed[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # Let's compute M[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # = sum_r (U[i, r] * V[j, r]) * W[k, r]
    # For fixed i, j: let x_r = U[i, r] * V[j, r], then M[i, j, :] = x @ W.T
    # So M[i, j, k] = sum_r x_r * W[k, r] = (x @ W.T)[k]
    
    # Compute all at once:
    # M[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # = sum_r (U @ V.T)[i, j, r] ... no.
    
    # Better: use einsum
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Now build the true tensor
    T_true = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    residual = M - T_true
    return np.max(np.abs(residual))

def _check_rank_reduction(U, V, W, n, m, p):
    """
    Try to reduce the rank by 1 by fixing two matrices and solving for the third.
    """
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    T_slices_U = [T_true[i].flatten() for i in range(nm)]
    T_slices_V = [T_true[:, j].flatten() for j in range(mp)]
    T_slices_W = [T_true[:, :, k].flatten() for k in range(pn)]
    
    # 1. Fix V, W, solve U
    V_W_kr = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, rank)
    b_U = np.array(T_slices_U)
    
    for c in range(rank):
        idx = [i for i in range(rank) if i != c]
        V_W_reduced = V_W_kr[:, idx]
        
        try:
            X, _, _, _ = np.linalg.lstsq(V_W_reduced.T, b_U.T, rcond=None)
            X = X.T
        except:
            continue
        
        U_new = np.zeros((nm, rank))
        U_new[:, idx] = X
        
        if np.linalg.norm(b_U.T - V_W_reduced.T @ X.T) < 1e-9:
            res = _compute_residual(U_new, V, W, n, m, p)
            if res < 1e-9:
                return U_new[:, idx], V[:, idx], W[:, idx]

    # 2. Fix U, W, solve V
    U_W_kr = np.einsum('ir,kr->ikr', U, W).reshape(nm * pn, rank)
    b_V = np.array(T_slices_V)
    
    for c in range(rank):
        idx = [i for i in range(rank) if i != c]
        U_W_reduced = U_W_kr[:, idx]
        
        try:
            X, _, _, _ = np.linalg.lstsq(U_W_reduced.T, b_V.T, rcond=None)
            X = X.T
        except:
            continue
        
        V_new = np.zeros((mp, rank))
        V_new[:, idx] = X
        
        if np.linalg.norm(b_V.T - U_W_reduced.T @ X.T) < 1e-9:
            res = _compute_residual(U, V_new, W, n, m, p)
            if res < 1e-9:
                return U[:, idx], V_new[:, idx], W[:, idx]

    # 3. Fix U, V, solve W
    U_V_kr = np.einsum('ir,jr->ijr', U, V).reshape(nm * mp, rank)
    b_W = np.array(T_slices_W)
    
    for c in range(rank):
        idx = [i for i in range(rank) if i != c]
        U_V_reduced = U_V_kr[:, idx]
        
        try:
            X, _, _, _ = np.linalg.lstsq(U_V_reduced.T, b_W.T, rcond=None)
            X = X.T
        except:
            continue
        
        W_new = np.zeros((pn, rank))
        W_new[:, idx] = X
        
        if np.linalg.norm(b_W.T - U_V_reduced.T @ X.T) < 1e-9:
            res = _compute_residual(U, V, W_new, n, m, p)
            if res < 1e-9:
                return U[:, idx], V[:, idx], W_new[:, idx]

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # First, build the rank-32 solution
    U, V, W = _build_rank32()
    
    # Convert to lists for return
    def to_list(arr):
        return arr.tolist()
    
    # Try linear reduction
    if time.time() < deadline - 20:
        result = _check_rank_reduction(U, V, W, n, m, p)
        if result is not None:
            U31, V31, W31 = result
            return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
            
    # Fall back to rank-32
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
