"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify U, V, W decompose the 2x3x8 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    max_err_target = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a, b, c] - 1.0)
                if err > max_err_target:
                    max_err_target = err
    
    target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_mask[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target_mask])) if (~target_mask).any() else 0.0
    
    return max_err_target <= tol and non_target_err <= tol

def _round_to_half_int(arr):
    return np.round(np.array(arr, dtype=float) * 2) / 2.0

def _try_round_and_verify(U, V, W, tol=1e-5):
    U_h = _round_to_half_int(U)
    V_h = _round_to_half_int(V)
    W_h = _round_to_half_int(W)
    if _verify_decomposition(U_h.tolist(), V_h.tolist(), W_h.tolist(), tol=tol):
        return U_h.tolist(), V_h.tolist(), W_h.tolist()
    U_i = np.round(np.array(U, dtype=float))
    V_i = np.round(np.array(V, dtype=float))
    W_i = np.round(np.array(W, dtype=float))
    if _verify_decomposition(U_i.tolist(), V_i.tolist(), W_i.tolist(), tol=tol):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    return None

def _build_rank44():
    """Build the known rank-44 decomposition."""
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen rank-7
        U[0, col] += 1; U[4, col] += 1
        V[k0, col] += 1; V[8+k1, col] += 1
        W[2*k0, col] += 1; W[2*k1+1, col] += 1
        col += 1
        
        U[3, col] += 1; U[4, col] += 1
        V[k0, col] += 1
        W[2*k0+1, col] += 1; W[2*k1+1, col] += -1
        col += 1
        
        U[0, col] += 1
        V[k1, col] += 1; V[8+k1, col] += -1
        W[2*k1, col] += 1; W[2*k1+1, col] += 1
        col += 1
        
        U[4, col] += 1
        V[8+k0, col] += 1; V[k0, col] += -1
        W[2*k0, col] += 1; W[2*k0+1, col] += 1
        col += 1
        
        U[0, col] += 1; U[1, col] += 1
        V[8+k1, col] += 1
        W[2*k0, col] += -1; W[2*k1, col] += 1
        col += 1
        
        U[3, col] += 1; U[0, col] += -1
        V[k0, col] += 1; V[k1, col] += 1
        W[2*k1+1, col] += 1
        col += 1
        
        U[1, col] += 1; U[4, col] += -1
        V[8+k0, col] += 1; V[8+k1, col] += 1
        W[2*k0, col] += 1
        col += 1
        
        # 4 rank-1 terms for A[:,2] * B[2,k0:k1]
        for ii in range(2):
            for kk in range(2):
                k = k0 if kk == 0 else k1
                U[ii*3+2, col] += 1
                V[16+k, col] += 1
                W[k*2+ii, col] += 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _attempt_als_decomposition(T, target_rank, n_iter=50, n_trials=10, time_limit=60.0):
    """
    Attempt to find a rank-target_rank decomposition of tensor T via ALS.
    Returns (U, V, W) as lists if successful, else None.
    """
    n, m, p = DIMENSIONS
    dims = (n*m, m*p, p*n)
    start = time.time()
    
    for trial in range(n_trials):
        if time.time() - start > time_limit:
            break
        
        rng = np.random.RandomState(trial * 1000 + 42)
        U_r = rng.randn(dims[0], target_rank) * 0.5
        V_r = rng.randn(dims[1], target_rank) * 0.5
        W_r = rng.randn(dims[2], target_rank) * 0.5
        
        for it in range(n_iter):
            if time.time() - start > time_limit:
                break
            
            # Update U_r
            VW = np.einsum('br,cr->bcr', V_r, W_r)
            VW_flat = VW.reshape(dims[1]*dims[2], target_rank)
            for a in range(dims[0]):
                T_flat = T[a, :, :].flatten()
                U_r[a, :], _, _, _ = np.linalg.lstsq(VW_flat, T_flat, rcond=None)
            
            # Update V_r
            UW = np.einsum('ar,cr->acr', U_r, W_r)
            UW_flat = UW.reshape(dims[0]*dims[2], target_rank)
            for b in range(dims[1]):
                T_flat = T[:, b, :].flatten()
                V_r[b, :], _, _, _ = np.linalg.lstsq(UW_flat, T_flat, rcond=None)
            
            # Update W_r
            UV = np.einsum('ar,br->abr', U_r, V_r)
            UV_flat = UV.reshape(dims[0]*dims[1], target_rank)
            for c in range(dims[2]):
                T_flat = T[:, :, c].flatten()
                W_r[c, :], _, _, _ = np.linalg.lstsq(UV_flat, T_flat, rcond=None)
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        err = np.max(np.abs(T_recon - T))
        
        if err < 1e-6:
            # Check if close to half-integers
            U_close = np.max(np.abs(U_r - _round_to_half_int(U_r)))
            V_close = np.max(np.abs(V_r - _round_to_half_int(V_r)))
            W_close = np.max(np.abs(W_r - _round_to_half_int(W_r)))
            if U_close < 1e-4 and V_close < 1e-4 and W_close < 1e-4:
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def construct():
    """Construct a rank-minimal decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Build the known rank-44 solution
    U44, V44, W44 = _build_rank44()
    assert _verify_decomposition(U44, V44, W44, tol=1e-9), "Rank-44 construction failed"
    
    best_rank = 44
    best_result = (U44, V44, W44)
    
    # Build the target tensor
    U_arr = np.array(U44, dtype=float)
    V_arr = np.array(V44, dtype=float)
    W_arr = np.array(W44, dtype=float)
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Try to find lower-rank decompositions
    start = time.time()
    time_budget = 100.0  # Safety margin for 150s limit
    
    for target_rank in range(43, 35, -1):
        if time.time() - start > time_budget:
            break
        
        result = _attempt_als_decomposition(T, target_rank, n_iter=40, n_trials=5, 
                                             time_limit=min(20.0, time_budget - (time.time() - start)))
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_decomposition(U_r, V_r, W_r, tol=1e-6):
                best_rank = target_rank
                best_result = (U_r, V_r, W_r)
                break
    
    U, V, W = best_result
    
    # Final safety check
    if not _verify_decomposition(U, V, W, tol=1e-8):
        U, V, W = U44, V44, W44
    
    return {"U": U, "V": V, "W": W}
