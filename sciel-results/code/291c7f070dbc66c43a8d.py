import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin

def _try_direct_rank22():
    """
    Try to directly construct a rank-22 decomposition for 2x3 x 3x4 matrix multiplication.
    Based on known tensor decomposition techniques.
    """
    n, m, p = 2, 3, 4
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    # We need to find U (6x22), V (12x22), W (8x22) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = 1 if (a,b,c) corresponds to valid (i,j,k)
    
    # Strategy: Start from schoolbook and try to merge columns
    # Schoolbook has 24 columns, we need to reduce to 22
    
    # Try using a structured approach: some columns can be shared
    # This is a heuristic construction
    
    rng = np.random.RandomState(42)
    
    # Initialize with schoolbook-like structure
    U = np.zeros((dim_U, 22))
    V = np.zeros((dim_V, 22))
    W = np.zeros((dim_W, 22))
    
    # First 20 columns: standard pattern with some sharing
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= 22:
                    break
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    # Try to fill remaining 2 columns with combinations
    # This is a rough heuristic - may not work directly
    if col < 22:
        for r in range(col, 22):
            # Try to create a column that covers multiple products
            U[:, r] = rng.randn(dim_U) * 0.1
            V[:, r] = rng.randn(dim_V) * 0.1
            W[:, r] = rng.randn(dim_W) * 0.1
    
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Construct target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Compute reconstructed tensor
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _als_decompose(T, R, n_iter=200, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    T: shape (A, B, C)
    R: rank
    Returns U, V, W or None if it fails to converge.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize with multiple strategies
    best_result = None
    best_err = float('inf')
    
    # Strategy 1: Small random
    U = rng.randn(A, R) * 0.1
    V = rng.randn(B, R) * 0.1
    W = rng.randn(C, R) * 0.1
    
    for it in range(n_iter):
        # Solve for U
        A_U = np.zeros((B * C, R))
        for r in range(R):
            A_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
        
        for a in range(A):
            b_a = T[a, :, :].flatten()
            u, _, _, _ = np.linalg.lstsq(A_U, b_a, rcond=None)
            U[a, :] = u
            
        # Solve for V
        A_V = np.zeros((A * C, R))
        for r in range(R):
            A_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
        for b in range(B):
            b_b = T[:, b, :].flatten()
            v, _, _, _ = np.linalg.lstsq(A_V, b_b, rcond=None)
            V[b, :] = v
            
        # Solve for W
        A_W = np.zeros((A * B, R))
        for r in range(R):
            A_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
        for c in range(C):
            b_c = T[:, :, c].flatten()
            w, _, _, _ = np.linalg.lstsq(A_W, b_c, rcond=None)
            W[c, :] = w
            
        # Check convergence
        if it % 5 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            if err < 1e-8:
                return U, V, W
                
    # Final check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) < 1e-6:
        return U, V, W
    
    if best_result is not None and best_err < 1e-4:
        return best_result
    
    return None

def _try_rank_als(rank, n, m, p, max_time=5, seed=42):
    """Try to find a rank-rank decomposition via ALS."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    start_time = time.time()
    
    # Try multiple initialization strategies
    for init_seed in [seed, seed + 100, seed + 200]:
        if time.time() - start_time > max_time - 1:
            break
            
        res = _als_decompose(T_target, rank, n_iter=150, seed=init_seed)
        
        if res is not None:
            U, V, W = res
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                
            # Try integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                
            # Try with smaller rounding
            for factor in [4, 8]:
                U_r = np.round(U * factor) / factor
                V_r = np.round(V * factor) / factor
                W_r = np.round(W * factor) / factor
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist())
            
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # First try direct construction for rank 22
    if time.time() < _DEADLINE - 20:
        res = _try_direct_rank22()
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
    
    # Try ranks from 22 to 24
    # 22 is the conjectured minimum, 24 is schoolbook
    for rank in [22, 23, 24]:
        if time.time() > _DEADLINE - 10:
            break
            
        # Try multiple seeds
        # Increase seeds for lower ranks to improve chance of finding the global min
        num_seeds = 30 if rank < 24 else 5
        for seed in range(num_seeds):
            if time.time() > _DEADLINE - 2:
                break
            res = _try_rank_als(rank, n, m, p, seed=seed)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
                
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
