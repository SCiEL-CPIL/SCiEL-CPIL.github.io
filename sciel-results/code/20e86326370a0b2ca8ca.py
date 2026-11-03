"""
Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.

Strategy:
1. Build a valid block decomposition using Strassen for 2x2x2 blocks.
2. Use alternating least squares with rank reduction to find a lower-rank
   decomposition that still exactly decomposes the multiplication tensor.
3. Use rank-1 peeling and residual deflation as an alternative initialization.
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
    Update U given V, W, and target tensor T.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    M = np.zeros((mp * pn, r), dtype=np.float64)
    for b in range(mp):
        for c in range(pn):
            M[b * pn + c, :] = V[b, :] * W[c, :]
    
    U_new = np.zeros_like(U)
    for a in range(nm):
        t_a = T_target[a, :, :].reshape(-1)
        MtM = M.T @ M + lam * np.eye(r)
        Mtt = M.T @ t_a
        try:
            U_new[a, :] = np.linalg.solve(MtM, Mtt)
        except np.linalg.LinAlgError:
            U_new[a, :] = np.linalg.lstsq(M, t_a, rcond=None)[0]
    
    return U_new

def _als_update_V(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update V given U, W, and target tensor T.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = V.shape[1]
    
    M = np.zeros((nm * pn, r), dtype=np.float64)
    for a in range(nm):
        for c in range(pn):
            M[a * pn + c, :] = U[a, :] * W[c, :]
    
    V_new = np.zeros_like(V)
    for b in range(mp):
        t_b = T_target[:, b, :].reshape(-1)
        MtM = M.T @ M + lam * np.eye(r)
        Mtt = M.T @ t_b
        try:
            V_new[b, :] = np.linalg.solve(MtM, Mtt)
        except np.linalg.LinAlgError:
            V_new[b, :] = np.linalg.lstsq(M, t_b, rcond=None)[0]
    
    return V_new

def _als_update_W(U, V, W, T_target, n, m, p, lam=0.01):
    """
    Update W given U, V, and target tensor T.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = W.shape[1]
    
    M = np.zeros((nm * mp, r), dtype=np.float64)
    for a in range(nm):
        for b in range(mp):
            M[a * mp + b, :] = U[a, :] * V[b, :]
    
    W_new = np.zeros_like(W)
    for c in range(pn):
        t_c = T_target[:, :, c].reshape(-1)
        MtM = M.T @ M + lam * np.eye(r)
        Mtt = M.T @ t_c
        try:
            W_new[c, :] = np.linalg.solve(MtM, Mtt)
        except np.linalg.LinAlgError:
            W_new[c, :] = np.linalg.lstsq(M, t_c, rcond=None)[0]
    
    return W_new

def _peel_rank1_residual(T_residual, n, m, p, deadline=None):
    """
    Peel off the best rank-1 component from the residual tensor using SVD of
    the best matrix unfolding. Returns (u, v, w, T_new_residual).
    """
    nm, mp, pn = n * m, m * p, p * n
    M = T_residual.reshape(nm, mp * pn)
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    if len(s) == 0 or s[0] == 0:
        return None
    
    u = U_svd[:, 0] * s[0]
    vw = Vt[0, :].reshape(mp, pn)
    
    U2, s2, V2t = np.linalg.svd(vw, full_matrices=False)
    if len(s2) == 0 or s2[0] == 0:
        return None
    v = U2[:, 0] * s2[0]
    w = V2t[0, :].copy()
    
    T_rank1 = np.einsum('a,b,c->abc', u, v, w, optimize=True)
    T_new = T_residual - T_rank1
    
    return u, v, w, T_new

def _try_reduce_rank_peel(U, V, W, n, m, p, target_rank, max_iters=50, deadline=None, tol=1e-9):
    """
    Try to reduce the rank to target_rank using rank-1 peeling followed by ALS.
    """
    T_target = _build_target_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    T_residual = T_target.copy()
    
    for step in range(target_rank - 1):
        if deadline is not None and time.time() > deadline - 15:
            return None
        
        result = _peel_rank1_residual(T_residual, n, m, p, deadline=deadline)
        if result is None:
            return None
        
        u, v, w, T_residual = result
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    U = np.hstack([U, np.zeros((nm, 1))])
    V = np.hstack([V, np.zeros((mp, 1))])
    W = np.hstack([W, np.zeros((pn, 1))])
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        U = _als_update_U(U, V, W, T_target, n, m, p, lam=0.0001)
        V = _als_update_V(U, V, W, T_target, n, m, p, lam=0.0001)
        W = _als_update_W(U, V, W, T_target, n, m, p, lam=0.0001)
        
        if _verify_decomp(n, m, p, U, V, W, tol):
            return U, V, W
    
    if _verify_decomp(n, m, p, U, V, W, tol):
        return U, V, W
    
    return None

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=50, deadline=None, tol=1e-9):
    """
    Try to reduce the rank to target_rank using ALS.
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

def _try_reduce_rank_multi_init(n, m, p, target_rank, n_inits=3, max_iters=50, deadline=None, tol=1e-9):
    """
    Try multiple random initializations for ALS rank reduction.
    """
    T_target = _build_target_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    rng = np.random.default_rng(42)
    
    best_result = None
    best_err = np.inf
    
    for init_idx in range(n_inits):
        if deadline is not None and time.time() > deadline - 20:
            break
        
        U = rng.standard_normal((nm, target_rank)) * 0.1
        V = rng.standard_normal((mp, target_rank)) * 0.1
        W = rng.standard_normal((pn, target_rank)) * 0.1
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 10:
                break
            
            U = _als_update_U(U, V, W, T_target, n, m, p, lam=0.001)
            V = _als_update_V(U, V, W, T_target, n, m, p, lam=0.001)
            W = _als_update_W(U, V, W, T_target, n, m, p, lam=0.001)
            
            if _verify_decomp(n, m, p, U, V, W, tol):
                return U, V, W
        
        # Compute error
        T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
        err = np.linalg.norm(T_comp - T_target)
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
    
    return best_result if best_result is not None and _verify_decomp(n, m, p, best_result[0], best_result[1], best_result[2], tol) else None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Step 1: Build the block decomposition (rank 84)
    U, V, W = _block_decomposition_356()
    current_rank = U.shape[1]
    
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
    
    # Step 2: Try to reduce rank using ALS and rank-1 peeling
    for target_rank in range(60, current_rank + 1, -5):
        if time.time() > deadline - 30:
            break
        
        result = _try_reduce_rank(U, V, W, n, m, p, target_rank, 
                                   max_iters=100, deadline=deadline)
        if result is None and time.time() < deadline - 25:
            result = _try_reduce_rank_peel(U, V, W, n, m, p, target_rank,
                                            max_iters=80, deadline=deadline)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = target_rank
            U, V, W = U_r, V_r, W_r
        else:
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
    
    # Step 3: Fine-tune around the best rank found
    if best_rank > 60 and time.time() < deadline - 30:
        for target_rank in range(max(55, best_rank - 5), best_rank + 1):
            if time.time() > deadline - 15:
                break
            
            result = _try_reduce_rank(best_U, best_V, best_W, n, m, p, 
                                       target_rank, max_iters=50, 
                                       deadline=deadline)
            if result is None and time.time() < deadline - 12:
                result = _try_reduce_rank_peel(best_U, best_V, best_W, n, m, p,
                                                target_rank, max_iters=40,
                                                deadline=deadline)
            if result is not None:
                U_r, V_r, W_r = result
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
    
    # Step 4: Try multi-init ALS for lower ranks
    if best_rank > 55 and time.time() < deadline - 40:
        for target_rank in range(max(50, best_rank - 10), best_rank):
            if time.time() > deadline - 20:
                break
            
            result = _try_reduce_rank_multi_init(n, m, p, target_rank, 
                                                  n_inits=3, max_iters=60,
                                                  deadline=deadline)
            if result is not None:
                U_r, V_r, W_r = result
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
