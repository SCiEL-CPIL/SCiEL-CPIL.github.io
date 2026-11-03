import numpy as np
import time

DIMENSIONS = [3, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for the (i,j,k) triple."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    expected = _build_tensor(n, m, p)
    return np.allclose(T, expected, atol=tol)

def _naive_decomposition(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1],
    ]
    V = [
        [1, 1, 0, -1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return U, V, W

def _block_decomposition(n, m, p, n_split, m_split, p_split):
    """Decompose n×m×p using specified block splits. Strassen for 2×2×2, schoolbook otherwise."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    for ii, ni_size in enumerate(n_split):
        for jj, mj_size in enumerate(m_split):
            for kk, pk_size in enumerate(p_split):
                i_off = sum(n_split[:ii])
                j_off = sum(m_split[:jj])
                k_off = sum(p_split[:kk])
                
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    U_s, V_s, W_s = _strassen_222()
                    for r in range(7):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for a in range(2):
                            for b in range(2):
                                u_col[(i_off + a) * m + (j_off + b)] = U_s[a * 2 + b][r]
                        for b in range(2):
                            for c in range(2):
                                v_col[(j_off + b) * p + (k_off + c)] = V_s[b * 2 + c][r]
                        for c in range(2):
                            for a in range(2):
                                w_col[(k_off + c) * n + (i_off + a)] = W_s[c * 2 + a][r]
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                else:
                    block_rank = ni_size * mj_size * pk_size
                    for a in range(ni_size):
                        for b in range(mj_size):
                            for c in range(pk_size):
                                u_col = [0.0] * (n * m)
                                v_col = [0.0] * (m * p)
                                w_col = [0.0] * (p * n)
                                u_col[(i_off + a) * m + (j_off + b)] = 1.0
                                v_col[(j_off + b) * p + (k_off + c)] = 1.0
                                w_col[(k_off + c) * n + (i_off + a)] = 1.0
                                U_cols.append(u_col)
                                V_cols.append(v_col)
                                W_cols.append(w_col)
    
    total_rank = len(U_cols)
    U = [[U_cols[r][a] for r in range(total_rank)] for a in range(n * m)]
    V = [[V_cols[r][a] for r in range(total_rank)] for a in range(m * p)]
    W = [[W_cols[r][a] for r in range(total_rank)] for a in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def _try_strassen_block_decomposition(n, m, p):
    """Try multiple split strategies and return the best valid one."""
    # Candidate split strategies: (n_split, m_split, p_split)
    strategies = [
        ([2,1], [2,2], [2,2,2,1]),  # 78
        ([2,1], [2,2], [2,2,1,2]),  # same count, different order
        ([2,1], [2,2], [1,2,2,2]),  # same count
        ([2,1], [2,2], [2,1,2,2]),  # has 2×2×1 blocks
        ([1,2], [2,2], [2,2,2,1]),  # same structure, n reversed
        ([2,1], [2,1,1], [2,2,2,1]), # m split differently
        ([2,1], [1,1,2], [2,2,2,1]), # m split differently
        ([2,1], [2,2], [3,2,2]),     # try 3 in p
        ([2,1], [3,1], [2,2,2,1]),   # try 3 in m
        ([3], [2,2], [2,2,2,1]),     # no n split
        ([2,1], [4], [2,2,2,1]),     # no m split
        ([2,1], [2,2], [4,3]),       # large p blocks
        ([2,1], [2,2], [3,4]),       # large p blocks
        ([2,1], [4], [3,4]),         # large blocks
    ]
    
    best_result = None
    best_rank = n * m * p
    
    for n_split, m_split, p_split in strategies:
        if sum(n_split) != n or sum(m_split) != m or sum(p_split) != p:
            continue
        if time.time() > _START_TIME + TIME_LIMIT - 30:
            break
        try:
            result = _block_decomposition(n, m, p, n_split, m_split, p_split)
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                rank = len(result["U"][0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        except Exception:
            pass
    
    if best_result is None:
        best_result = _naive_decomposition(n, m, p)
    
    return best_result

def _compress_decomposition(U, V, W, n, m, p, target_rank, max_iters=500, tol=1e-10):
    """
    Try to compress a rank-r decomposition to rank-r' < r by iteratively
    zeroing out components and solving for a valid lower-rank factorization.
    
    Uses the approach: given a valid rank-r decomposition, we want to find
    a valid rank-r' decomposition. We do this by starting with the rank-r
    solution and trying to eliminate one rank term at a time.
    """
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return None
    
    # Build the tensor from current decomposition
    T = np.zeros((n * m, m * p, p * n))
    for r in range(current_rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Try to reduce rank by trying to eliminate terms one at a time
    # Strategy: for each term, try to express it as a linear combination
    # of the others (in the tensor space)
    
    # Alternative approach: use random projections to find a lower-rank
    # representation. We project the rank-r decomposition onto a random
    # subspace of dimension target_rank and solve.
    
    # Method: Given rank-r factors (U, V, W), find rank-r' factors (U', V', W')
    # such that sum_r U_r ⊗ V_r ⊗ W_r = sum_{r'} U'_r' ⊗ V'_r' ⊗ W'_r'
    
    # Use the fact that if we have a valid rank-r decomposition, we can try
    # to find a valid rank-r' decomposition by solving a system of equations.
    
    # For each target rank, try random initialization and solve
    rng = np.random.default_rng(42)
    
    for attempt in range(max_iters):
        if time.time() > _START_TIME + TIME_LIMIT - 20:
            break
        
        # Random initialization for the lower-rank factors
        U_new = rng.standard_normal((n * m, target_rank)) * 0.1
        V_new = rng.standard_normal((m * p, target_rank)) * 0.1
        W_new = rng.standard_normal((p * n, target_rank)) * 0.1
        
        # Iteratively update: fix two, solve for the third
        # This is like alternating least squares
        converged = False
        for it in range(100):
            # Update U: for each column i, solve min ||T - sum_r U[i,r]*V[:,r]*W[:,r]||
            # This is linear in U[:,r] for fixed V, W
            # T[i,:,:] = sum_r U[i,r] * (V[:,r] ⊗ W[:,r])
            # So for each i: T[i,:,:] = sum_r U[i,r] * vec(V[:,r] ⊗ W[:,r])
            # Flatten: T_flat[i, :] = U[i, :] @ M_vw where M_vw[r, (j,k)] = V[j,r]*W[k,r]
            
            # Build the matrix M_vw of shape (target_rank, m*p*p*n)
            # M_vw[r, j*(p*n) + k] = V[j, r] * W[k, r]
            M_vw = np.zeros((target_rank, m * p * p * n))
            for r in range(target_rank):
                for j in range(m * p):
                    for k in range(p * n):
                        M_vw[r, j * (p * n) + k] = V_new[j, r] * W_new[k, r]
            
            # T_flat[i, :] is the i-th slice of T flattened
            T_flat = T.reshape(n * m, m * p * p * n)
            
            # Solve U_new[i, :] @ M_vw.T = T_flat[i, :]
            # i.e., M_vw @ U_new[i, :].T = T_flat[i, :].T
            # Use least squares
            try:
                U_new = np.linalg.lstsq(M_vw, T_flat.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                break
            
            # Update V: for each column j, solve
            # T[:, j, :] = sum_r U[:, r] * V[j, r] * W[:, r]
            # For fixed U, W: T[:, j, k] = sum_r V[j, r] * (U[:, r] * W[k, r])
            # M_uw[r, i*(p*n) + k] = U[i, r] * W[k, r]
            M_uw = np.zeros((target_rank, n * m * p * n))
            for r in range(target_rank):
                for i in range(n * m):
                    for k in range(p * n):
                        M_uw[r, i * (p * n) + k] = U_new[i, r] * W_new[k, r]
            
            T_flat2 = T.reshape(m * p, n * m * p * n)
            
            try:
                V_new = np.linalg.lstsq(M_uw, T_flat2.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                break
            
            # Update W: for each column k, solve
            # T[:, :, k] = sum_r U[:, r] * V[:, r] * W[k, r]
            # For fixed U, V: T[i, j, k] = sum_r W[k, r] * (U[i, r] * V[j, r])
            # M_uv[r, i*(m*p) + j] = U[i, r] * V[j, r]
            M_uv = np.zeros((target_rank, n * m * m * p))
            for r in range(target_rank):
                for i in range(n * m):
                    for j in range(m * p):
                        M_uv[r, i * (m * p) + j] = U_new[i, r] * V_new[j, r]
            
            T_flat3 = T.reshape(p * n, n * m * m * p)
            
            try:
                W_new = np.linalg.lstsq(M_uv, T_flat3.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                break
            
            # Check convergence
            T_check = np.zeros((n * m, m * p, p * n))
            for r in range(target_rank):
                T_check += np.einsum('a,b,c->abc', U_new[:, r], V_new[:, r], W_new[:, r])
            if np.allclose(T_check, T, atol=tol):
                converged = True
                break
        
        if converged:
            # Round to nearest half-integers for cleanliness
            U_out = np.round(U_new * 2) / 2
            V_out = np.round(V_new * 2) / 2
            W_out = np.round(W_new * 2) / 2
            
            # Verify with rounded values
            T_round = np.zeros((n * m, m * p, p * n))
            for r in range(target_rank):
                T_round += np.einsum('a,b,c->abc', U_out[:, r], V_out[:, r], W_out[:, r])
            if np.allclose(T_round, T, atol=1e-6):
                return {
                    "U": U_out.tolist(),
                    "V": V_out.tolist(),
                    "W": W_out.tolist()
                }
            else:
                # Use unrounded values
                return {
                    "U": U_new.tolist(),
                    "V": V_new.tolist(),
                    "W": W_new.tolist()
                }
    
    return None

def _try_compression(n, m, p, base_result):
    """Try to compress the base decomposition to a lower rank."""
    U = np.array(base_result["U"], dtype=np.float64)
    V = np.array(base_result["V"], dtype=np.float64)
    W = np.array(base_result["W"], dtype=np.float64)
    current_rank = U.shape[1]
    
    best_result = base_result
    best_rank = current_rank
    
    # Try target ranks from current_rank - 1 down to some minimum
    # The theoretical lower bound for 3x4x4x7 is unknown but at least max(3*4, 4*7, 3*7) = 28
    # In practice, likely higher. Try a range of targets.
    
    for target_rank in range(current_rank - 1, max(20, current_rank - 15), -1):
        if time.time() > _START_TIME + TIME_LIMIT - 25:
            break
        
        compressed = _compress_decomposition(U, V, W, n, m, p, target_rank, max_iters=50)
        if compressed is not None:
            if _verify(compressed["U"], compressed["V"], compressed["W"], n, m, p, tol=1e-6):
                rank = len(compressed["U"][0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = compressed
                # If we found a valid compression, try even lower
                # but don't go below the found rank
                if rank < target_rank:
                    # Got lucky, continue trying lower
                    pass
                else:
                    # Found exactly target_rank, try one lower
                    continue
            else:
                continue
        else:
            continue
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + TIME_LIMIT
    
    best_result = _try_strassen_block_decomposition(n, m, p)
    
    # Try to compress the best block decomposition
    if time.time() < deadline - 30:
        compressed = _try_compression(n, m, p, best_result)
        if compressed is not None:
            comp_rank = len(compressed["U"][0])
            base_rank = len(best_result["U"][0])
            if comp_rank < base_rank:
                best_result = compressed
    
    return best_result
