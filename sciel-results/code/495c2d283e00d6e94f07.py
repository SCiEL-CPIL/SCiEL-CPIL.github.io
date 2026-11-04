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

def _round_to_halfint(arr):
    """Round to nearest half-integer."""
    rounded = np.round(arr * 2) / 2
    return rounded

def _greedy_column_removal(U, V, W, deadline):
    """
    Try to remove columns one at a time if the remaining columns
    can still represent the tensor (checked via least-squares fit).
    """
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    # Get the target tensor
    U3 = U.reshape(n, m, current_rank)
    V3 = V.reshape(m, p, current_rank)
    W3 = W.reshape(p, n, current_rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Try removing each column
    for trial in range(20):
        if time.time() > deadline:
            break
        
        removed = False
        for col_idx in range(U.shape[1]):
            if time.time() > deadline:
                break
            
            # Skip this column
            mask = [i for i in range(U.shape[1]) if i != col_idx]
            U_sub = U[:, mask]
            V_sub = V[:, mask]
            W_sub = W[:, mask]
            
            # Check if the sub-decomposition already equals T
            err = _check_decomposition_fast(U_sub, V_sub, W_sub)
            if err < 1e-10:
                # Column is redundant
                U = U_sub
                V = V_sub
                W = W_sub
                removed = True
                break
        
        if not removed:
            break
    
    return U, V, W

def _vectorized_als(U, V, W, T, target_rank, max_iters, deadline):
    """
    Vectorized ALS for tensor decomposition.
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    """
    n, m, p = DIMENSIONS
    Ur = U.copy()
    Vr = V.copy()
    Wr = W.copy()
    
    best_err = float('inf')
    best_U, best_V, best_W = Ur.copy(), Vr.copy(), Wr.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve M_ij @ U_ij = T_ij
        # M_ij[k, r] = V[j,k,r] * W[k,i,r]
        # T_ij[k] = sum_r M_ij[k,r] * U_ij[r]
        
        U_new = np.zeros((n * m, target_rank))
        for i in range(n):
            for j in range(m):
                # M shape: (p, target_rank)
                M = Vr[j * p:(j+1)*p, :].T * Wr.reshape(p, n, target_rank)[:, i, :]
                # M[k, r] = V[j,k,r] * W[k,i,r]
                # Solve M.T @ x = T[i,j,:]
                b = T[i, j, :]
                # M.T is (target_rank, p), b is (p,)
                try:
                    x, _, _, _ = np.linalg.lstsq(M.T, b, rcond=None)
                except:
                    x = np.zeros(target_rank)
                U_new[i * m + j, :] = x
        
        Ur = U_new
        
        # Update V: for each (j,k), solve N_jk @ V_jk = T[:,j,k]
        # N_jk[i, r] = U[i,j,r] * W[k,i,r]
        V_new = np.zeros((m * p, target_rank))
        for j in range(m):
            for k in range(p):
                # N shape: (n, target_rank)
                N = Ur.reshape(n, m, target_rank)[:, j, :].T * Wr[k * n:(k+1)*n, :].T
                # N[i, r] = U[i,j,r] * W[k,i,r]
                b = T[:, j, k]
                try:
                    x, _, _, _ = np.linalg.lstsq(N.T, b, rcond=None)
                except:
                    x = np.zeros(target_rank)
                V_new[j * p + k, :] = x
        
        Vr = V_new
        
        # Update W: for each (k,i), solve P_ki @ W_ki = T[i,:,k]
        # P_ki[j, r] = U[i,j,r] * V[j,k,r]
        W_new = np.zeros((p * n, target_rank))
        for k in range(p):
            for i in range(n):
                # P shape: (m, target_rank)
                P = Ur.reshape(n, m, target_rank)[i, :, :].T * Vr.reshape(m, p, target_rank)[:, k, :].T
                # P[j, r] = U[i,j,r] * V[j,k,r]
                b = T[i, :, k]
                try:
                    x, _, _, _ = np.linalg.lstsq(P.T, b, rcond=None)
                except:
                    x = np.zeros(target_rank)
                W_new[k * n + i, :] = x
        
        Wr = W_new
        
        # Check error
        U3 = Ur.reshape(n, m, target_rank)
        V3 = Vr.reshape(m, p, target_rank)
        W3 = Wr.reshape(p, n, target_rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.max(np.abs(T_recon - T))
        
        if err < best_err:
            best_err = err
            best_U = Ur.copy()
            best_V = Vr.copy()
            best_W = Wr.copy()
        
        if err < 1e-10:
            break
    
    return best_U, best_V, best_W, best_err

def _rank_reduction_via_als(U, V, W, target_rank, deadline):
    """
    Try to reduce rank using vectorized ALS.
    """
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W, 0.0
    
    # Get the target tensor
    U3 = U.reshape(n, m, current_rank)
    V3 = V.reshape(m, p, current_rank)
    W3 = W.reshape(p, n, current_rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    rng = np.random.RandomState(42)
    
    num_trials = 5
    iters_per_trial = 200
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Initialize with random low-rank factors, scaled
        scale = 1.0 / np.sqrt(target_rank)
        Ur = rng.randn(n * m, target_rank) * scale
        Vr = rng.randn(m * p, target_rank) * scale
        Wr = rng.randn(p * n, target_rank) * scale
        
        Ur_r, Vr_r, Wr_r, err = _vectorized_als(Ur, Vr, Wr, T, target_rank, iters_per_trial, deadline)
        
        if err < best_err:
            best_err = err
            best_U = Ur_r
            best_V = Vr_r
            best_W = Wr_r
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-10:
        return best_U, best_V, best_W, best_err
    else:
        return None, None, None, best_err

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Build the initial rank-55 decomposition
    U, V, W = _build_strassen_decomp()
    
    err = _check_decomposition_fast(U, V, W)
    if err > 1e-6:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 55
    best_result = (U.copy(), V.copy(), W.copy())
    
    # First, try greedy column removal
    U, V, W = _greedy_column_removal(U, V, W, deadline)
    best_rank = U.shape[1]
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Try to reduce rank further using ALS
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 10:
            break
        
        Ur, Vr, Wr, err = _rank_reduction_via_als(U, V, W, target_rank, deadline)
        
        if Ur is not None and err < 1e-10:
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
