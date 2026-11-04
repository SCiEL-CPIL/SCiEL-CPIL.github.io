import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _build_strassen_decomp():
    """Build the rank-55 Strassen-based decomposition as numpy arrays."""
    n, m, p = DIMENSIONS
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    W_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            r = col
            for local_idx, val in L[s].items():
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j, r] = val
            for local_idx, val in R[s].items():
                j = local_idx // 2
                k = local_idx % 2
                V[j * p + (k_offset + k), r] = val
            for local_idx, val in W_coeffs[s].items():
                i = local_idx // 2
                k = local_idx % 2
                W[(k_offset + k) * n + i, r] = val
            col += 1
    
    for i in range(2):
        for k in range(10):
            r = col
            U[i * m + 2, r] = 1
            V[2 * p + k, r] = 1
            W[k * n + i, r] = 1
            col += 1
    
    return U, V, W

def _check_decomposition_fast(U, V, W):
    """Fast check using einsum."""
    n, m, p = DIMENSIONS
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    max_err = np.max(np.abs(T - 1.0))
    return max_err

def _als_decomposition(U, V, W, target_rank, max_iters=200, tol=1e-10, deadline=None):
    """ALS to find a lower-rank decomposition using vectorized updates."""
    n, m, p = DIMENSIONS
    
    # Build the target tensor (all ones for matmul)
    T = np.ones((n, m, p))
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    # Try multiple random initializations
    n_inits = 5
    for init in range(n_inits):
        if deadline is not None and time.time() > deadline:
            break
            
        rng = np.random.RandomState(42 + init * 7)
        Ur = rng.randn(n * m, target_rank) * 0.1
        Vr = rng.randn(m * p, target_rank) * 0.1
        Wr = rng.randn(p * n, target_rank) * 0.1
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve M[i,j,:]^T @ u = T[i,j,:]
            # M[i,j,k] = sum_r Vr[j*p+k,r] * Wr[k*n+i,r]
            Vr3 = Vr.reshape(m, p, target_rank)
            Wr3 = Wr.reshape(p, n, target_rank)
            M = np.einsum('jkr,kir->ijk', Vr3, Wr3)  # (n, m, p)
            
            U_new = np.zeros((n * m, target_rank))
            for i in range(n):
                for j in range(m):
                    Mij = M[i, j, :]  # (p,)
                    Tij = T[i, j, :]  # (p,)
                    # Solve: Mij^T @ u = Tij, i.e., Mij @ u.T = Tij.T
                    # This is a (p x target_rank) system
                    try:
                        u, _, _, _ = np.linalg.lstsq(Mij, Tij, rcond=None)
                    except:
                        u = np.zeros(target_rank)
                    U_new[i * m + j, :] = u
            
            # Update V
            Ur3 = Ur.reshape(n, m, target_rank)
            Wr3 = Wr.reshape(p, n, target_rank)
            N = np.einsum('ijr,kir->jki', Ur3, Wr3)  # (m, p, n)
            
            V_new = np.zeros((m * p, target_rank))
            for j in range(m):
                for k in range(p):
                    Njk = N[j, k, :]  # (n,)
                    Tjk = T[:, j, k]  # (n,)
                    try:
                        v, _, _, _ = np.linalg.lstsq(Njk, Tjk, rcond=None)
                    except:
                        v = np.zeros(target_rank)
                    V_new[j * p + k, :] = v
            
            # Update W
            Ur3 = Ur.reshape(n, m, target_rank)
            Vr3 = Vr.reshape(m, p, target_rank)
            P = np.einsum('ijr,jkr->kij', Ur3, Vr3)  # (p, n, m)
            
            W_new = np.zeros((p * n, target_rank))
            for k in range(p):
                for i in range(n):
                    Pki = P[k, i, :]  # (m,)
                    Tki = T[i, :, k]  # (m,)
                    try:
                        w, _, _, _ = np.linalg.lstsq(Pki, Tki, rcond=None)
                    except:
                        w = np.zeros(target_rank)
                    W_new[k * n + i, :] = w
            
            Ur, Vr, Wr = U_new, V_new, W_new
            
            # Check error
            Ur3 = Ur.reshape(n, m, target_rank)
            Vr3 = Vr.reshape(m, p, target_rank)
            Wr3 = Wr.reshape(p, n, target_rank)
            T_recon = np.einsum('ijr,jkr,kir->ijk', Ur3, Vr3, Wr3)
            err = np.max(np.abs(T_recon - T))
            
            if err < best_err:
                best_err = err
                best_U = Ur.copy()
                best_V = Vr.copy()
                best_W = Wr.copy()
            
            if err < tol:
                break
    
    return best_U, best_V, best_W, best_err

def _round_to_halfint(arr):
    """Round to nearest half-integer."""
    return np.round(arr * 2) / 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the initial rank-55 decomposition
    U, V, W = _build_strassen_decomp()
    
    # Verify it's correct
    err = _check_decomposition_fast(U, V, W)
    if err > 1e-6:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 55
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Try to reduce rank using ALS
    # Start from rank 40 down to 30, with fewer iterations for higher ranks
    for target_rank in range(45, 29, -1):
        if time.time() > deadline:
            break
        
        # Use more iterations for lower target ranks (harder)
        iters = 50 if target_rank > 35 else 80
        
        Ur, Vr, Wr, err = _als_decomposition(U, V, W, target_rank, max_iters=iters, tol=1e-10, deadline=deadline)
        
        if Ur is not None and err < 1e-8:
            # Try to round to half-integers
            Ur_r = _round_to_halfint(Ur)
            Vr_r = _round_to_halfint(Vr)
            Wr_r = _round_to_halfint(Wr)
            
            err_rounded = _check_decomposition_fast(Ur_r, Vr_r, Wr_r)
            if err_rounded < 1e-6:
                best_rank = target_rank
                best_result = (Ur_r, Vr_r, Wr_r)
            else:
                best_rank = target_rank
                best_result = (Ur, Vr, Wr)
            break
    
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
