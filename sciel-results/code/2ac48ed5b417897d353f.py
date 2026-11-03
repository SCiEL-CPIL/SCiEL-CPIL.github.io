"""
Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.

Strategy:
1. Build a valid block decomposition using Strassen for 2x2x2 blocks.
2. Use alternating least squares with rank reduction to find a lower-rank
   decomposition that still exactly decomposes the multiplication tensor.
3. Add sparse linear algebra refinement: use L1-regularized ALS to sparsify
   the factor matrices, then attempt to round to half-integers and verify.
4. Return the best valid decomposition found within the time budget.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_comp, T_target, atol=tol)

def _schoolbook(n_sub, m_sub, p_sub):
    """Return schoolbook decomposition for n_sub x m_sub x p_sub."""
    rank = n_sub * m_sub * p_sub
    U = np.zeros((n_sub * m_sub, rank), dtype=np.float64)
    V = np.zeros((m_sub * p_sub, rank), dtype=np.float64)
    W = np.zeros((p_sub * n_sub, rank), dtype=np.float64)
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U[i * m_sub + j, col] = 1.0
                V[j * p_sub + k, col] = 1.0
                W[k * n_sub + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7. Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed_sub_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_off, j_off, k_off, n, m, p):
    """
    Embed a sub-decomposition into the full 3x5x6 problem.
    """
    rank = U_sub.shape[1]
    U_full = np.zeros((n * m, rank), dtype=np.float64)
    V_full = np.zeros((m * p, rank), dtype=np.float64)
    W_full = np.zeros((p * n, rank), dtype=np.float64)
    
    for idx in range(n_sub * m_sub):
        i_s = idx // m_sub
        j_s = idx % m_sub
        i_f = i_off + i_s
        j_f = j_off + j_s
        U_full[i_f * m + j_f, :] = U_sub[idx, :]
    
    for idx in range(m_sub * p_sub):
        j_s = idx // p_sub
        k_s = idx % p_sub
        j_f = j_off + j_s
        k_f = k_off + k_s
        V_full[j_f * p + k_f, :] = V_sub[idx, :]
    
    for idx in range(p_sub * n_sub):
        k_s = idx // n_sub
        i_s = idx % n_sub
        k_f = k_off + k_s
        i_f = i_off + i_s
        W_full[k_f * n + i_f, :] = W_sub[idx, :]
    
    return U_full, V_full, W_full

def _block_decomposition_356():
    """
    Construct a block decomposition for 3x5x6 with rank 84.
    Split: n=[2,1], m=[2,2,1], p=[2,2,2]
    Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.
    """
    n, m, p = 3, 5, 6
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_total = np.zeros((n * m, 0), dtype=np.float64)
    V_total = np.zeros((m * p, 0), dtype=np.float64)
    W_total = np.zeros((p * n, 0), dtype=np.float64)
    
    i_off = 0
    for i_s, n_sub in enumerate(n_splits):
        j_off = 0
        for j_s, m_sub in enumerate(m_splits):
            k_off = 0
            for k_s, p_sub in enumerate(p_splits):
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    U_sub, V_sub, W_sub = _strassen_222()
                else:
                    U_sub, V_sub, W_sub = _schoolbook(n_sub, m_sub, p_sub)
                
                U_full, V_full, W_full = _embed_sub_decomp(
                    U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                    i_off, j_off, k_off, n, m, p
                )
                
                if U_total.shape[1] == 0:
                    U_total = U_full
                    V_total = V_full
                    W_total = W_full
                else:
                    U_total = np.hstack([U_total, U_full])
                    V_total = np.hstack([V_total, V_full])
                    W_total = np.hstack([W_total, W_full])
                
                k_off += p_sub
            j_off += m_sub
        i_off += n_sub
    
    return U_total, V_total, W_total

def _als_update_U(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update U given V, W, and target tensor T. Vectorized.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    # M has shape (mp*pn, r): M[b*pn+c, :] = V[b,:] * W[c,:]
    M = np.einsum('bk,ck->bck', V, W).reshape(mp * pn, r)
    
    # Solve for each row of U: U[a,:] = (M^T M + lam*I)^{-1} M^T t_a
    MtM = M.T @ M + lam * np.eye(r)
    try:
        MtM_inv = np.linalg.inv(MtM)
    except np.linalg.LinAlgError:
        MtM_inv = np.linalg.pinv(MtM)
    
    # t_a = T_target[a,:,:].reshape(-1) for each a
    # U_new[a,:] = MtM_inv @ (M.T @ t_a)
    # = MtM_inv @ (M.T @ T_target[a,:,:].reshape(-1))
    # Vectorize: U_new = (T_target @ M) @ MtM_inv  -- but shapes don't match directly
    # T_target is (nm, mp, pn), M is (mp*pn, r)
    # M.T @ t_a for all a: (T_target.reshape(nm, -1) @ M.T).T = M @ T_target.reshape(nm,-1).T
    # Actually: for each a, M.T @ t_a where t_a = T_target[a].reshape(-1)
    # Stack: (mp*pn, nm) = M @ T_target.reshape(nm, mp*pn).T ... no
    # M is (mp*pn, r), t_a is (mp*pn,)
    # M.T @ t_a is (r,)
    # For all a: M.T @ T_target.reshape(nm, mp*pn).T gives (r, nm)
    rhs = M.T @ T_target.reshape(nm, mp * pn).T  # (r, nm)
    U_new = (MtM_inv @ rhs).T  # (nm, r)
    
    return U_new

def _als_update_V(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update V given U, W, and target tensor T. Vectorized.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = V.shape[1]
    
    # M has shape (nm*pn, r): M[a*pn+c, :] = U[a,:] * W[c,:]
    M = np.einsum('ak,ck->ack', U, W).reshape(nm * pn, r)
    
    MtM = M.T @ M + lam * np.eye(r)
    try:
        MtM_inv = np.linalg.inv(MtM)
    except np.linalg.LinAlgError:
        MtM_inv = np.linalg.pinv(MtM)
    
    # For each b: t_b = T_target[:,b,:].reshape(-1), shape (nm*pn,)
    # M.T @ t_b: (r,)
    # T_target[:,b,:].reshape(-1) = T_target.transpose(1,0,2).reshape(mp, nm*pn)[b]
    rhs = M.T @ T_target.transpose(1, 0, 2).reshape(mp, nm * pn).T  # (r, mp)
    V_new = (MtM_inv @ rhs).T  # (mp, r)
    
    return V_new

def _als_update_W(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update W given U, V, and target tensor T. Vectorized.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = W.shape[1]
    
    # M has shape (nm*mp, r): M[a*mp+b, :] = U[a,:] * V[b,:]
    M = np.einsum('ak,bk->abk', U, V).reshape(nm * mp, r)
    
    MtM = M.T @ M + lam * np.eye(r)
    try:
        MtM_inv = np.linalg.inv(MtM)
    except np.linalg.LinAlgError:
        MtM_inv = np.linalg.pinv(MtM)
    
    # For each c: t_c = T_target[:,:,c].reshape(-1), shape (nm*mp,)
    # T_target[:,:,c].reshape(-1) = T_target.transpose(2,0,1).reshape(pn, nm*mp)[c]
    rhs = M.T @ T_target.transpose(2, 0, 1).reshape(pn, nm * mp).T  # (r, pn)
    W_new = (MtM_inv @ rhs).T  # (pn, r)
    
    return W_new

def _sparse_als_update_U(U, V, W, T_target, n, m, p, lam=0.01, l1_weight=0.01):
    """
    Sparse ALS update for U with L1 regularization (soft-thresholding). Vectorized.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    M = np.einsum('bk,ck->bck', V, W).reshape(mp * pn, r)
    MtM = M.T @ M + lam * np.eye(r)
    try:
        MtM_inv = np.linalg.inv(MtM)
    except np.linalg.LinAlgError:
        MtM_inv = np.linalg.pinv(MtM)
    
    rhs = M.T @ T_target.reshape(nm, mp * pn).T  # (r, nm)
    sol = (MtM_inv @ rhs).T  # (nm, r)
    U_new = np.sign(sol) * np.maximum(np.abs(sol) - l1_weight, 0.0)
    
    return U_new

def _sparse_als_update_V(U, V, W, T_target, n, m, p, lam=0.01, l1_weight=0.01):
    """
    Sparse ALS update for V with L1 regularization (soft-thresholding). Vectorized.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = V.shape[1]
    
    M = np.einsum('ak,ck->ack', U, W).reshape(nm * pn, r)
    MtM = M.T @ M + lam * np.eye(r)
    try:
        MtM_inv = np.linalg.inv(MtM)
    except np.linalg.LinAlgError:
        MtM_inv = np.linalg.pinv(MtM)
    
    rhs = M.T @ T_target.transpose(1, 0, 2).reshape(mp, nm * pn).T  # (r, mp)
    sol = (MtM_inv @ rhs).T  # (mp, r)
    V_new = np.sign(sol) * np.maximum(np.abs(sol) - l1_weight, 0.0)
    
    return V_new

def _sparse_als_update_W(U, V, W, T_target, n, m, p, lam=0.01, l1_weight=0.01):
    """
    Sparse ALS update for W with L1 regularization (soft-thresholding). Vectorized.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = W.shape[1]
    
    M = np.einsum('ak,bk->abk', U, V).reshape(nm * mp, r)
    MtM = M.T @ M + lam * np.eye(r)
    try:
        MtM_inv = np.linalg.inv(MtM)
    except np.linalg.LinAlgError:
        MtM_inv = np.linalg.pinv(MtM)
    
    rhs = M.T @ T_target.transpose(2, 0, 1).reshape(pn, nm * mp).T  # (r, pn)
    sol = (MtM_inv @ rhs).T  # (pn, r)
    W_new = np.sign(sol) * np.maximum(np.abs(sol) - l1_weight, 0.0)
    
    return W_new

def _round_to_half_int(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _try_sparse_refinement(U, V, W, n, m, p, deadline, tol=1e-9):
    """
    Try sparse ALS refinement to find a sparser (and potentially lower-rank)
    decomposition. Returns (U, V, W) if successful, else None.
    """
    T_target = _build_target_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = r
    best_err = float('inf')
    
    # Try sparse ALS with various L1 weights
    for l1 in [0.001, 0.01, 0.05, 0.1]:
        if time.time() > deadline - 20:
            break
        
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        
        for it in range(30):
            if time.time() > deadline - 15:
                break
            
            U_curr = _sparse_als_update_U(U_curr, V_curr, W_curr, T_target, n, m, p, lam=0.001, l1_weight=l1)
            V_curr = _sparse_als_update_V(U_curr, V_curr, W_curr, T_target, n, m, p, lam=0.001, l1_weight=l1)
            W_curr = _sparse_als_update_W(U_curr, V_curr, W_curr, T_target, n, m, p, lam=0.001, l1_weight=l1)
            
            # Check if decomposition is still valid
            if _verify_decomp(n, m, p, U_curr, V_curr, W_curr, tol):
                # Try to reduce rank by removing zero columns
                col_norms = np.max(np.abs(np.vstack([U_curr, V_curr, W_curr])), axis=0)
                nonzero_mask = col_norms > 1e-10
                if np.sum(nonzero_mask) < r:
                    U_r = U_curr[:, nonzero_mask]
                    V_r = V_curr[:, nonzero_mask]
                    W_r = W_curr[:, nonzero_mask]
                    if _verify_decomp(n, m, p, U_r, V_r, W_r, tol):
                        if U_r.shape[1] < best_rank:
                            best_rank = U_r.shape[1]
                            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                            best_err = 0.0
                            return best_U, best_V, best_W
                else:
                    if _verify_decomp(n, m, p, U_curr, V_curr, W_curr, tol):
                        if r < best_rank:
                            best_rank = r
                            best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                            best_err = 0.0
        
        # After sparse ALS, try rounding to half-integers
        U_rounded = _round_to_half_int(U_curr)
        V_rounded = _round_to_half_int(V_curr)
        W_rounded = _round_to_half_int(W_curr)
        
        if _verify_decomp(n, m, p, U_rounded, V_rounded, W_rounded, tol):
            # Try to reduce rank
            col_norms = np.max(np.abs(np.vstack([U_rounded, V_rounded, W_rounded])), axis=0)
            nonzero_mask = col_norms > 1e-10
            if np.sum(nonzero_mask) < r:
                U_r = U_rounded[:, nonzero_mask]
                V_r = V_rounded[:, nonzero_mask]
                W_r = W_rounded[:, nonzero_mask]
                if _verify_decomp(n, m, p, U_r, V_r, W_r, tol):
                    if U_r.shape[1] < best_rank:
                        best_rank = U_r.shape[1]
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                        best_err = 0.0
                        return best_U, best_V, best_W
            else:
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
                    best_err = 0.0
    
    if best_err < float('inf') and best_rank < r:
        return best_U, best_V, best_W
    
    return None

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=50, deadline=None, tol=1e-9):
    """
    Try to reduce the rank to target_rank using ALS.
    Returns (U, V, W) if successful, else None.
    """
    T_target = _build_target_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    
    if U.shape[1] < target_rank:
        diff = target_rank - U.shape[1]
        U = np.hstack([U, np.zeros((nm, diff))])
        V = np.hstack([V, np.zeros((mp, diff))])
        W = np.hstack([W, np.zeros((pn, diff))])
    else:
        U = U[:, :target_rank].copy()
        V = V[:, :target_rank].copy()
        W = W[:, :target_rank].copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        U = _als_update_U(U, V, W, T_target, n, m, p, lam=0.001)
        V = _als_update_V(U, V, W, T_target, n, m, p, lam=0.001)
        W = _als_update_W(U, V, W, T_target, n, m, p, lam=0.001)
        
        if _verify_decomp(n, m, p, U, V, W, tol):
            return U, V, W
    
    if _verify_decomp(n, m, p, U, V, W, tol):
        return U, V, W
    
    return None

def _try_column_elimination(U, V, W, n, m, p, tol=1e-9, deadline=None):
    """
    Try to eliminate columns that are linearly dependent on the others.
    For each column index j, check if column j of the "effective" factor
    can be expressed as a linear combination of the other columns.
    If so, remove it and verify.
    
    Strategy: For each column j, we check if the rank of the full decomposition
    is preserved when we zero out column j and re-solve for the remaining factors.
    A simpler approach: check if the columns of U, V, W are linearly dependent
    in a way that allows elimination.
    
    More directly: try removing each column and see if the remaining r-1 columns
    can still represent the tensor (via a quick ALS check with few iterations).
    """
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    T_target = _build_target_tensor(n, m, p)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = r
    
    # Try removing each column one at a time
    for j in range(r):
        if deadline is not None and time.time() > deadline - 15:
            break
        
        mask = np.ones(r, dtype=bool)
        mask[j] = False
        r_new = r - 1
        if r_new < 1:
            break
        
        U_try = U[:, mask].copy()
        V_try = V[:, mask].copy()
        W_try = W[:, mask].copy()
        
        # Quick ALS refinement (few iterations)
        for it in range(20):
            if deadline is not None and time.time() > deadline - 10:
                break
            U_try = _als_update_U(U_try, V_try, W_try, T_target, n, m, p, lam=0.001)
            V_try = _als_update_V(U_try, V_try, W_try, T_target, n, m, p, lam=0.001)
            W_try = _als_update_W(U_try, V_try, W_try, T_target, n, m, p, lam=0.001)
        
        if _verify_decomp(n, m, p, U_try, V_try, W_try, tol):
            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
            best_rank = r_new
            # Recurse: try to eliminate more columns
            if deadline is not None and time.time() < deadline - 20:
                result = _try_column_elimination(best_U, best_V, best_W, n, m, p, tol, deadline)
                if result is not None:
                    return result
            return best_U, best_V, best_W
    
    return None

def _high_precision_polish(U, V, W, n, m, p, deadline, tol=1e-12):
    """
    Final high-precision polish pass with tighter tolerance and more iterations.
    """
    T_target = _build_target_tensor(n, m, p)
    r = U.shape[1]
    
    for it in range(50):
        if time.time() > deadline - 5:
            break
        
        U = _als_update_U(U, V, W, T_target, n, m, p, lam=0.0001)
        V = _als_update_V(U, V, W, T_target, n, m, p, lam=0.0001)
        W = _als_update_W(U, V, W, T_target, n, m, p, lam=0.0001)
        
        if _verify_decomp(n, m, p, U, V, W, tol):
            return U, V, W
    
    if _verify_decomp(n, m, p, U, V, W, tol):
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Step 1: Build the block decomposition (rank 84)
    U, V, W = _block_decomposition_356()
    current_rank = U.shape[1]
    
    # Verify initial decomposition
    if not _verify_decomp(n, m, p, U, V, W, tol=1e-8):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Step 2: Try to reduce rank using ALS
    for target_rank in range(60, current_rank + 1, -5):
        if time.time() > deadline - 40:
            break
        
        result = _try_reduce_rank(U, V, W, n, m, p, target_rank, 
                                   max_iters=100, deadline=deadline)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = target_rank
            U, V, W = U_r, V_r, W_r
        else:
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
    
    # Step 3: Sparse linear algebra refinement
    if time.time() < deadline - 30:
        sparse_result = _try_sparse_refinement(best_U, best_V, best_W, n, m, p, deadline)
        if sparse_result is not None:
            U_s, V_s, W_s = sparse_result
            if U_s.shape[1] < best_rank:
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = U_s.shape[1]
    
    # Step 4: Column elimination (direct rank reduction)
    if time.time() < deadline - 30:
        elim_result = _try_column_elimination(best_U, best_V, best_W, n, m, p, 
                                               deadline=deadline)
        if elim_result is not None:
            U_e, V_e, W_e = elim_result
            if U_e.shape[1] < best_rank:
                best_U, best_V, best_W = U_e, V_e, W_e
                best_rank = U_e.shape[1]
    
    # Step 5: Fine-tune around the best rank found
    if best_rank > 60 and time.time() < deadline - 20:
        for target_rank in range(max(55, best_rank - 5), best_rank + 1):
            if time.time() > deadline - 10:
                break
            
            result = _try_reduce_rank(best_U, best_V, best_W, n, m, p, 
                                       target_rank, max_iters=50, 
                                       deadline=deadline)
            if result is not None:
                U_r, V_r, W_r = result
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
    
    # Step 6: High-precision polish
    if time.time() < deadline - 5:
        polished = _high_precision_polish(best_U, best_V, best_W, n, m, p, deadline)
        if polished is not None:
            best_U, best_V, best_W = polished
    
    # Final verification
    if not _verify_decomp(n, m, p, best_U, best_V, best_W, tol=1e-8):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
