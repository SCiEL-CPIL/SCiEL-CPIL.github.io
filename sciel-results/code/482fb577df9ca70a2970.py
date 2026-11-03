import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                target[a, b, c] = 1.0
    decomp = np.einsum('it,jt,kt->ijk', U, V, W)
    diff = np.abs(decomp - target)
    return np.max(diff) < tol

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen222():
    # Rank-7 Strassen decomposition of the 2x2x2 matrix multiplication tensor.
    # Term t contributes: sum over (a,b,c) of u_t[a]*v_t[b]*w_t[c] * e_a e_b e_c
    # where a = i*2+k (A row i, col k), b = k*2+j (B row k, col j), c = j*2+i (C col j, row i)
    terms = []
    A00, A01, A10, A11 = 0, 1, 2, 3
    B00, B01, B10, B11 = 0, 1, 2, 3
    C00, C01, C10, C11 = 0, 1, 2, 3

    M = {
        'M1': ([(A00, 1), (A11, 1)], [(B00, 1), (B11, 1)]),
        'M2': ([(A10, 1), (A11, 1)], [(B00, 1)]),
        'M3': ([(A00, 1)], [(B01, 1), (B11, -1)]),
        'M4': ([(A11, 1)], [(B10, 1), (B00, -1)]),
        'M5': ([(A00, 1), (A10, 1)], [(B10, 1)]),
        'M6': ([(A11, 1), (A01, -1)], [(B00, 1), (B01, 1)]),
        'M7': ([(A01, 1), (A11, -1)], [(B10, 1), (B11, 1)]),
    }

    C_contrib = {
        'M1': [(C00, 1), (C11, 1)],
        'M2': [(C10, 1), (C11, -1)],
        'M3': [(C01, 1), (C11, 1)],
        'M4': [(C00, 1), (C10, 1)],
        'M5': [(C00, -1), (C01, 1)],
        'M6': [(C11, 1)],
        'M7': [(C00, 1)],
    }

    for name in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']:
        ua, ub = M[name]
        u = [0.0] * 4
        for idx, c in ua:
            u[idx] = c
        v = [0.0] * 4
        for idx, c in ub:
            v[idx] = c
        w = [0.0] * 4
        for idx, c in C_contrib[name]:
            w[idx] = c
        terms.append((u, v, w))
    return terms

def _build_target(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[a, b, c] = 1.0
    return T

def _block_decomposition_356():
    """
    Construct a block decomposition for 3x5x6.
    Split: n=[2,1], m=[2,2,1], p=[2,2,2]
    Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.
    Returns (U, V, W) as numpy arrays.
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
                    strassen = _strassen222()
                    r_sub = len(strassen)
                    U_sub = np.zeros((n_sub * m_sub, r_sub), dtype=np.float64)
                    V_sub = np.zeros((m_sub * p_sub, r_sub), dtype=np.float64)
                    W_sub = np.zeros((p_sub * n_sub, r_sub), dtype=np.float64)
                    for col, (u, v, w) in enumerate(strassen):
                        for i in range(2):
                            for k in range(2):
                                U_sub[i * 2 + k, col] = u[i * 2 + k]
                        for k in range(2):
                            for j in range(2):
                                V_sub[k * 2 + j, col] = v[k * 2 + j]
                        for j in range(2):
                            for i in range(2):
                                W_sub[j * 2 + i, col] = w[j * 2 + i]
                else:
                    r_sub = n_sub * m_sub * p_sub
                    U_sub = np.zeros((n_sub * m_sub, r_sub), dtype=np.float64)
                    V_sub = np.zeros((m_sub * p_sub, r_sub), dtype=np.float64)
                    W_sub = np.zeros((p_sub * n_sub, r_sub), dtype=np.float64)
                    col = 0
                    for i in range(n_sub):
                        for k in range(m_sub):
                            for j in range(p_sub):
                                U_sub[i * m_sub + k, col] = 1.0
                                V_sub[k * p_sub + j, col] = 1.0
                                W_sub[j * n_sub + i, col] = 1.0
                                col += 1
                
                # Embed into full dimensions
                U_full = np.zeros((n * m, r_sub), dtype=np.float64)
                V_full = np.zeros((m * p, r_sub), dtype=np.float64)
                W_full = np.zeros((p * n, r_sub), dtype=np.float64)
                
                for idx in range(n_sub * m_sub):
                    i_s_idx = idx // m_sub
                    j_s_idx = idx % m_sub
                    i_f = i_off + i_s_idx
                    j_f = j_off + j_s_idx
                    U_full[i_f * m + j_f, :] = U_sub[idx, :]
                
                for idx in range(m_sub * p_sub):
                    j_s_idx = idx // p_sub
                    k_s_idx = idx % p_sub
                    j_f = j_off + j_s_idx
                    k_f = k_off + k_s_idx
                    V_full[j_f * p + k_f, :] = V_sub[idx, :]
                
                for idx in range(p_sub * n_sub):
                    k_s_idx = idx // n_sub
                    i_s_idx = idx % n_sub
                    k_f = k_off + k_s_idx
                    i_f = i_off + i_s_idx
                    W_full[k_f * n + i_f, :] = W_sub[idx, :]
                
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

def _als_step(U, V, W, T, n, m, p, lam=1e-6, which='U'):
    """
    One ALS update step. Updates U, V, or W depending on 'which'.
    """
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    if which == 'U':
        M = np.zeros((mp * pn, r), dtype=np.float64)
        for b in range(mp):
            for c in range(pn):
                M[b * pn + c, :] = V[b, :] * W[c, :]
        U_new = np.zeros_like(U)
        for a in range(nm):
            t_a = T[a, :, :].reshape(-1)
            MtM = M.T @ M + lam * np.eye(r)
            Mtt = M.T @ t_a
            try:
                U_new[a, :] = np.linalg.solve(MtM, Mtt)
            except np.linalg.LinAlgError:
                U_new[a, :] = np.linalg.lstsq(M, t_a, rcond=None)[0]
        return U_new, V, W
    elif which == 'V':
        M = np.zeros((nm * pn, r), dtype=np.float64)
        for a in range(nm):
            for c in range(pn):
                M[a * pn + c, :] = U[a, :] * W[c, :]
        V_new = np.zeros_like(V)
        for b in range(mp):
            t_b = T[:, b, :].reshape(-1)
            MtM = M.T @ M + lam * np.eye(r)
            Mtt = M.T @ t_b
            try:
                V_new[b, :] = np.linalg.solve(MtM, Mtt)
            except np.linalg.LinAlgError:
                V_new[b, :] = np.linalg.lstsq(M, t_b, rcond=None)[0]
        return U, V_new, W
    elif which == 'W':
        M = np.zeros((nm * mp, r), dtype=np.float64)
        for a in range(nm):
            for b in range(mp):
                M[a * mp + b, :] = U[a, :] * V[b, :]
        W_new = np.zeros_like(W)
        for c in range(pn):
            t_c = T[:, :, c].reshape(-1)
            MtM = M.T @ M + lam * np.eye(r)
            Mtt = M.T @ t_c
            try:
                W_new[c, :] = np.linalg.solve(MtM, Mtt)
            except np.linalg.LinAlgError:
                W_new[c, :] = np.linalg.lstsq(M, t_c, rcond=None)[0]
        return U, V, W_new
    else:
        raise ValueError("which must be 'U', 'V', or 'W'")

def _try_reduce_rank(U, V, W, T, n, m, p, target_rank, max_iters=50, tol=1e-9, deadline=None):
    """
    Try to reduce rank to target_rank using ALS with iterative rank peeling.
    Returns (U, V, W) with rank target_rank if successful, else None.
    """
    nm, mp, pn = n * m, m * p, p * n
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Start from current decomposition
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    # We want to reduce from current_rank to target_rank.
    # Strategy: iteratively try to set one column to zero by re-solving.
    # This is heuristic and may not always work.
    
    # A more robust approach: use ALS to refine the decomposition,
    # then try to identify and remove redundant components.
    
    # For now, we'll use a simple iterative approach:
    # 1. Run ALS for a few iterations to converge.
    # 2. Try to remove columns with small norm.
    
    for iter in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Run one ALS cycle
        U_cur, V_cur, W_cur = _als_step(U_cur, V_cur, W_cur, T, n, m, p, which='U')
        U_cur, V_cur, W_cur = _als_step(U_cur, V_cur, W_cur, T, n, m, p, which='V')
        U_cur, V_cur, W_cur = _als_step(U_cur, V_cur, W_cur, T, n, m, p, which='W')
        
        # Check if we can remove any columns
        if U_cur.shape[1] > target_rank:
            # Find columns with small norm
            norms = np.linalg.norm(U_cur, axis=0)
            min_norm_idx = np.argmin(norms)
            min_norm = norms[min_norm_idx]
            
            if min_norm < 1e-8:
                # Remove this column
                mask = np.ones(U_cur.shape[1], dtype=bool)
                mask[min_norm_idx] = False
                U_cur = U_cur[:, mask]
                V_cur = V_cur[:, mask]
                W_cur = W_cur[:, mask]
                
                if U_cur.shape[1] == target_rank:
                    # Verify
                    decomp = np.einsum('it,jt,kt->ijk', U_cur, V_cur, W_cur)
                    if np.max(np.abs(decomp - T)) < tol:
                        return U_cur, V_cur, W_cur
    
    # If we couldn't reduce to target_rank, try a direct approach:
    # Initialize a new decomposition of rank target_rank and fit it.
    U_new = np.zeros((nm, target_rank), dtype=np.float64)
    V_new = np.zeros((mp, target_rank), dtype=np.float64)
    W_new = np.zeros((pn, target_rank), dtype=np.float64)
    
    # Initialize with random values
    np.random.seed(42)
    U_new = np.random.randn(nm, target_rank)
    V_new = np.random.randn(mp, target_rank)
    W_new = np.random.randn(pn, target_rank)
    
    # Run ALS
    for iter in range(max_iters * 2):
        if deadline is not None and time.time() > deadline:
            break
        
        U_new, V_new, W_new = _als_step(U_new, V_new, W_new, T, n, m, p, which='U')
        U_new, V_new, W_new = _als_step(U_new, V_new, W_new, T, n, m, p, which='V')
        U_new, V_new, W_new = _als_step(U_new, V_new, W_new, T, n, m, p, which='W')
        
        # Check convergence
        decomp = np.einsum('it,jt,kt->ijk', U_new, V_new, W_new)
        err = np.max(np.abs(decomp - T))
        if err < tol:
            return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = 3, 5, 6
    deadline = time.time() + 130.0
    
    # Build block decomposition
    U, V, W = _block_decomposition_356()
    T = _build_target(n, m, p)
    
    # Verify block decomposition
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try to reduce rank
    # Start from current rank and try lower ranks
    for target_rank in range(best_rank - 1, 1, -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_reduce_rank(U, V, W, T, n, m, p, target_rank, 
                                  max_iters=20, deadline=deadline - 5)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = target_rank
                break
    
    # Convert to list of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Final verification
    if _verify(U_list, V_list, W_list, n, m, p):
        return result
    return _schoolbook(n, m, p)
