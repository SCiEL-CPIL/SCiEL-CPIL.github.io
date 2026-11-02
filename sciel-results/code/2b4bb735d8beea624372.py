import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# Precompute the target tensor T with entries T[i*m+j, j*p+k, k*n+i] = 1
n, m, p = DIMENSIONS
def _target_tensor():
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

def _verify(U, V, W):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    T = _get_T()
    # Compute reconstructed tensor: R_(a,b,c) = sum_r U[a,r] V[b,r] W[c,r]
    # Efficiently: U (nm x R), V (mp x R), W (pn x R)
    # For each r: outer product U[:,r], V[:,r], W[:,r]
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.abs(recon - T)
    return np.max(diff) < 1e-9

def _round_half_int(M, tol=1e-6):
    """Round a matrix to nearest half-integers if possible."""
    M2 = 2.0 * np.asarray(M, dtype=float)
    Mr = np.round(M2)
    if np.max(np.abs(M2 - Mr)) < tol:
        return Mr / 2.0, True
    return None, False

def _als_decompose(T, R, n, m, p, max_iter=200, tol=1e-7):
    """
    Alternating least squares to find U (nm x R), V (mp x R), W (pn x R)
    such that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c].
    Vectorized version for speed.
    """
    nm, mp, pn = n*m, m*p, p*n
    # Structured initialization: start with schoolbook and add small noise
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    # Initialize with a subset of schoolbook columns if R < nm*mp*pn
    schoolbook_cols = min(R, nm * mp * pn)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= schoolbook_cols:
                    break
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    # Add small random perturbation to avoid degenerate solutions
    U += np.random.randn(nm, R) * 0.01
    V += np.random.randn(mp, R) * 0.01
    W += np.random.randn(pn, R) * 0.01

    for it in range(max_iter):
        # Update U: vectorized least squares
        # For each a: solve min_X sum_{b,c} (sum_r X[r] V[b,r] W[c,r] - T[a,b,c])^2
        # This is: (V*W)^T (V*W) X = (V*W)^T T[a]
        # where (V*W)[(b,c),r] = V[b,r]*W[c,r]
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)  # (mp, pn, R)
        MW_flat = MW.reshape(mp * pn, R)
        # Compute Gram matrix and RHS for all a at once
        # Gram = MW_flat.T @ MW_flat (R x R)
        # For each a: rhs_a = MW_flat.T @ T[a].reshape(mp*pn)
        # U = solve(Gram, rhs_all) where rhs_all is (nm, R)
        T_reshaped = T.reshape(nm, mp * pn)
        # rhs_all[a, r] = sum_{b,c} T[a,b,c] * MW_flat[(b,c), r]
        rhs_all = T_reshaped @ MW_flat  # (nm, R)
        Gram = MW_flat.T @ MW_flat  # (R, R)
        # Add small regularization for numerical stability
        Gram += np.eye(R) * 1e-10
        try:
            U = np.linalg.solve(Gram, rhs_all.T).T
        except np.linalg.LinAlgError:
            U, _, _, _ = np.linalg.lstsq(Gram, rhs_all.T, rcond=None)
            U = U.T

        # Update V
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)  # (nm, pn, R)
        UW_flat = UW.reshape(nm * pn, R)
        T_reshaped_v = np.transpose(T, (1, 0, 2)).reshape(mp, nm * pn)
        rhs_all_v = T_reshaped_v @ UW_flat  # (mp, R)
        Gram_v = UW_flat.T @ UW_flat  # (R, R)
        Gram_v += np.eye(R) * 1e-10
        try:
            V = np.linalg.solve(Gram_v, rhs_all_v.T).T
        except np.linalg.LinAlgError:
            V, _, _, _ = np.linalg.lstsq(Gram_v, rhs_all_v.T, rcond=None)
            V = V.T

        # Update W
        UV = np.einsum('ar,br->abr', U, V, optimize=True)  # (nm, mp, R)
        UV_flat = UV.reshape(nm * mp, R)
        T_reshaped_w = np.transpose(T, (2, 0, 1)).reshape(pn, nm * mp)
        rhs_all_w = T_reshaped_w @ UV_flat  # (pn, R)
        Gram_w = UV_flat.T @ UV_flat  # (R, R)
        Gram_w += np.eye(R) * 1e-10
        try:
            W = np.linalg.solve(Gram_w, rhs_all_w.T).T
        except np.linalg.LinAlgError:
            W, _, _, _ = np.linalg.lstsq(Gram_w, rhs_all_w.T, rcond=None)
            W = W.T

        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T))
        if err < tol:
            break

    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(recon - T))
    if err < 1e-6:
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

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave margin from 150s limit
    n, m, p = DIMENSIONS
    T = _get_T()
    
    # Try ranks from low to high
    # Minimum possible rank is at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But actually the rank is at least the tensor rank. For matrix mult, known bounds.
    # Let's try from 30 up to 90 (schoolbook)
    for R in range(30, 91):
        if time.time() > deadline - 15.0:
            break
        
        # Try several random restarts
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            result = _als_decompose(T, R, n, m, p, max_iter=50, tol=1e-8)
            if result is not None:
                U_r, V_r, W_r = result
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                # Convert to integers if all are integers
                def to_int_or_half(lst):
                    out = []
                    for row in lst:
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
                return {"U": to_int_or_half(U_list), "V": to_int_or_half(V_list), "W": to_int_or_half(W_list)}
    
    # Fallback to schoolbook
    return _construct_schoolbook()
