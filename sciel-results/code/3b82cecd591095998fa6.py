import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _build_mm_tensor(n, m, p):
    """Build the MM tensor T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    T[a,b,c] = 1 if j==j', i==i', k==k' else 0
    """
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _validate(U, V, W, T, tol=1e-6):
    """Validate: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]"""
    result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(result, T, atol=tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _als_decomposition(T, rank, n_iter=500, tol=1e-10, time_limit=60.0, seed=42):
    """Alternating Least Squares to find rank-rank decomposition."""
    start_time = time.time()
    nm, mp, pn = T.shape
    rng = np.random.RandomState(seed)
    
    U = rng.randn(nm, rank) * 0.5
    V = rng.randn(mp, rank) * 0.5
    W = rng.randn(pn, rank) * 0.5
    
    prev_error = np.inf
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Update U
        VW = np.einsum('br,cr->bc', V, W, optimize=True)
        T_flat = T.reshape(nm, -1)
        try:
            U = np.linalg.lstsq(VW, T_flat.T, rcond=None)[0].T
        except:
            break
        
        # Update V
        UW = np.einsum('ar,cr->ac', U, W, optimize=True)
        T_reshape = T.transpose(1, 0, 2).reshape(mp, -1)
        try:
            V = np.linalg.lstsq(UW, T_reshape.T, rcond=None)[0].T
        except:
            break
        
        # Update W
        UV = np.einsum('ar,br->ab', U, V, optimize=True)
        T_reshape2 = T.transpose(2, 0, 1).reshape(pn, -1)
        try:
            W = np.linalg.lstsq(UV, T_reshape2.T, rcond=None)[0].T
        except:
            break
        
        result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        error = np.sum((result - T) ** 2)
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    return U, V, W, error

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    T = _build_mm_tensor(n, m, p)
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    assert _validate(U_sb, V_sb, W_sb, T), "Schoolbook validation failed"
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower-rank decomposition using ALS
    for target_rank in range(best_rank - 1, max(20, best_rank - 25)):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            U_r, V_r, W_r, error = _als_decomposition(
                T, target_rank, n_iter=400, time_limit=35.0, seed=seed
            )
            if error < 1e-6 and _validate(U_r, V_r, W_r, T, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
