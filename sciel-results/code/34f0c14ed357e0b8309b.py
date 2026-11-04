"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_fast(U, V, W, n, m, p, T_true=None):
    """Fast verification of tensor decomposition."""
    if T_true is None:
        T_true = _build_true_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T_true, atol=1e-9)

def _als_refine(U, V, W, T_true, n, m, p, max_iters=200, tol=1e-10):
    """ALS refinement for tensor decomposition. Returns (U, V, W, converged)."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    # Precompute T_true in flattened form for efficiency
    T_flat = T_true.reshape(nm, mp, pn)
    
    for it in range(max_iters):
        # Update U: for each a, solve U[a,:] @ M = T_true[a,:,:].reshape(-1)
        # M[b*pn+c, :] = V[b,:] * W[c,:]
        # Build M efficiently
        M = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, r)
        # Solve least squares
        rhs = T_flat.reshape(nm, mp*pn).T  # (mp*pn, nm)
        try:
            U_new, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
            U = U_new.T
        except:
            pass
        
        # Update V
        M_v = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, r)
        rhs_v = T_flat.reshape(nm, mp, pn).permute(1, 0, 2).reshape(mp, nm*pn).T
        try:
            V_new, _, _, _ = np.linalg.lstsq(M_v, rhs_v, rcond=None)
            V = V_new.T
        except:
            pass
        
        # Update W
        M_w = np.einsum('ar,br->abr', U, V).reshape(nm*mp, r)
        rhs_w = T_flat.reshape(nm, mp, pn).permute(2, 0, 1).reshape(pn, nm*mp).T
        try:
            W_new, _, _, _ = np.linalg.lstsq(M_w, rhs_w, rcond=None)
            W = W_new.T
        except:
            pass
        
        # Check convergence
        if it % 20 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T_true))
            if err < tol:
                return U, V, W, True
    
    # Final check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T_true))
    return U, V, W, err < 1e-8

def _try_rank_structured(r, n, m, p, T_true, deadline, seed=42):
    """Try to find a rank-r decomposition using structured initialization + ALS."""
    nm, mp, pn = n*m, m*p, p*n
    
    if time.time() > deadline:
        return None
    
    # Structured initialization based on block decomposition
    # Split j into j1 (high) and j2 (low), k into k1 (high) and k2 (low)
    # j = j1*2 + j2, k = k1*4 + k2
    
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    W = np.zeros((pn, r))
    
    # Try to assign rank terms based on structure
    # For each (i, j1, k1), create a "block" of rank terms
    # j1 in {0,1}, k1 in {0,1,2,3}
    
    # Initialize with a structured pattern
    col = 0
    for i in range(n):
        for j1 in range(2):  # j = j1*2 + j2, j2 in {0,1}
            for k1 in range(4):  # k = k1*4 + k2, k2 in {0,1,2,3}
                for j2 in range(2):
                    for k2 in range(4):
                        j = j1 * 2 + j2
                        k = k1 * 4 + k2
                        if col < r:
                            U[i*m + j, col] = 1.0
                            V[j*p + k, col] = 1.0
                            W[k*n + i, col] = 1.0
                            col += 1
                # If we've exhausted rank, stop
                if col >= r:
                    break
            if col >= r:
                break
        if col >= r:
            break
    
    # If we didn't fill all rank terms, use random for the rest
    if col < r:
        rng = np.random.RandomState(seed)
        for c in range(col, r):
            U[:, c] = rng.randn(nm) * 0.1
            V[:, c] = rng.randn(mp) * 0.1
            W[:, c] = rng.randn(pn) * 0.1
    
    # ALS refinement
    U, V, W, converged = _als_refine(U, V, W, T_true, n, m, p, max_iters=100, tol=1e-10)
    
    if converged:
        # Try to round to half-integers
        for scale in [1, 2, 4]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if _verify_fast(U_r, V_r, W_r, n, m, p, T_true):
                return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        # Return as-is if verification passes
        if _verify_fast(U, V, W, n, m, p, T_true):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def _try_rank_random(r, n, m, p, T_true, deadline, seed=42):
    """Try to find a rank-r decomposition using random initialization + ALS."""
    nm, mp, pn = n*m, m*p, p*n
    
    if time.time() > deadline:
        return None
    
    rng = np.random.RandomState(seed)
    
    for init in range(3):
        if time.time() > deadline:
            return None
        
        U = rng.randn(nm, r) * 0.1
        V = rng.randn(mp, r) * 0.1
        W = rng.randn(pn, r) * 0.1
        
        U, V, W, converged = _als_refine(U, V, W, T_true, n, m, p, max_iters=80, tol=1e-10)
        
        if converged:
            for scale in [1, 2, 4]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                if _verify_fast(U_r, V_r, W_r, n, m, p, T_true):
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist())
            if _verify_fast(U, V, W, n, m, p, T_true):
                return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 110  # 110 second budget to be safe
    
    T_true = _build_true_tensor(n, m, p)
    
    # Try candidate ranks in increasing order
    # Start from a reasonable lower bound
    candidates = [32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]
    
    for r in candidates:
        if time.time() > deadline - 15:
            break
        if r >= n * m * p:
            break
        
        # Try structured first
        result = _try_rank_structured(r, n, m, p, T_true, deadline, seed=42)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try random
        result = _try_rank_random(r, n, m, p, T_true, deadline, seed=42 + r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
