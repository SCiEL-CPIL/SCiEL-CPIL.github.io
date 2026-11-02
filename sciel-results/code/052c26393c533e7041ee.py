import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

_T_cache = None
def _get_T():
    global _T_cache
    if _T_cache is None:
        _T_cache = _target_tensor()
    return _T_cache

def _verify(U, V, W, tol=1e-8):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = _get_T()
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.abs(recon - T)
    return np.max(diff) < tol

def _round_half_int(M, tol=1e-6):
    """Round a matrix to nearest half-integers if possible."""
    M2 = 2.0 * np.asarray(M, dtype=float)
    Mr = np.round(M2)
    if np.max(np.abs(M2 - Mr)) < tol:
        return Mr / 2.0, True
    return None, False

def _als_decompose(T, R, n, m, p, max_iter=50, tol=1e-7, seed=None, time_limit=5.0):
    """
    Alternating least squares to find U (nm x R), V (mp x R), W (pn x R)
    such that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c].
    """
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    
    rng = np.random.RandomState(seed)
    U = rng.randn(nm, R) * 0.1
    V = rng.randn(mp, R) * 0.1
    W = rng.randn(pn, R) * 0.1
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Update U
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)
        MW_flat = MW.reshape(mp*pn, R)
        MW_T = MW_flat.T
        for a in range(nm):
            target_a = T[a].reshape(mp*pn)
            X, _, _, _ = np.linalg.lstsq(MW_T, target_a, rcond=None)
            U[a, :] = X
            
        # Update V
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)
        UW_flat = UW.reshape(nm*pn, R)
        UW_T = UW_flat.T
        for b in range(mp):
            target_b = T[:, b, :].reshape(nm*pn)
            X, _, _, _ = np.linalg.lstsq(UW_T, target_b, rcond=None)
            V[b, :] = X
            
        # Update W
        UV = np.einsum('ar,br->abr', U, V, optimize=True)
        UV_flat = UV.reshape(nm*mp, R)
        UV_T = UV_flat.T
        for c in range(pn):
            target_c = T[:, :, c].reshape(nm*mp)
            X, _, _, _ = np.linalg.lstsq(UV_T, target_c, rcond=None)
            W[c, :] = X
            
        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T))
        if err < tol:
            break
    
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(recon - T))
    if err < 1e-5:
        # Try to round to half-integers
        U_r, ok1 = _round_half_int(U, tol=1e-4)
        if ok1:
            V_r, ok2 = _round_half_int(V, tol=1e-4)
            if ok2:
                W_r, ok3 = _round_half_int(W, tol=1e-4)
                if ok3:
                    if _verify(U_r, V_r, W_r):
                        return U_r, V_r, W_r
    return None

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _to_list(M):
    """Convert numpy array to list of lists with integer/half-integer values."""
    M = np.asarray(M, dtype=float)
    out = []
    for row in M:
        out_row = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                out_row.append(int(round(val)))
            elif abs(val * 2 - round(val * 2)) < 1e-9:
                out_row.append(round(val * 2) / 2.0)
            else:
                out_row.append(val)
        out.append(out_row)
    return out

def construct():
    start_time = time.time()
    deadline = 140.0
    n, m, p = DIMENSIONS
    T = _get_T()
    
    # Lower bound for rank is max(n*m, m*p, p*n) = 30
    # Upper bound is n*m*p = 90
    # We search from low to high to find the minimal rank
    
    # We will allocate time to each rank.
    # There are 61 possible ranks (30 to 90).
    # Total time ~ 140s.
    # Let's give ~2 seconds per rank initially, more for promising ones?
    # Or just iterate and break if time is up.
    
    for R in range(30, 91):
        if time.time() > deadline:
            break
        
        # Try a few restarts for this rank
        # If R is close to 90, it's easier. If R is low, it's harder.
        n_restarts = 3 if R > 70 else 5
        found = False
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Time limit per ALS run
            # Leave some time for verification and overhead
            time_left = deadline - time.time()
            t_limit = min(5.0, time_left / (n_restarts - restart))
            if t_limit < 0.5:
                break
                
            result = _als_decompose(T, R, n, m, p, max_iter=30, tol=1e-7, seed=restart*100+R, time_limit=t_limit)
            if result is not None:
                U, V, W = result
                U_list = _to_list(U)
                V_list = _to_list(V)
                W_list = _to_list(W)
                return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback to schoolbook
    return _construct_schoolbook()
