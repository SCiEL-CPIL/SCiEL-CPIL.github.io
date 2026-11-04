import numpy as np
import time

def construct():
    """
    Rank-reduced decomposition for 4x4 * 4x8 matrix multiplication.
    
    Builds a rank-98 decomposition via Kronecker product of Strassen 2x2x2,
    then attempts rank reduction via tensor column linear dependencies.
    """
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s limit
    
    n, m, p = 4, 4, 8
    
    # Build the base rank-98 Strassen decomposition
    result = construct_rank98(n, m, p)
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    # Verify the base decomposition
    try:
        if not _verify_decomposition(U, V, W, n, m, p):
            return construct_naive(n, m, p)
    except Exception:
        return construct_naive(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > 80:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank via direct tensor column rank check
    try:
        rank_reduced = _reduce_rank_direct(U, V, W, n, m, p, deadline - start_time)
        if rank_reduced is not None:
            U_r, V_r, W_r = rank_reduced
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    except Exception:
        pass
    
    # Try additional reduction: exploit rectangular structure
    try:
        rank_reduced2 = _reduce_rank_rectangular(U, V, W, n, m, p, deadline - start_time)
        if rank_reduced2 is not None:
            U_r, V_r, W_r = rank_reduced2
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    except Exception:
        pass
    
    # Return the base decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct_rank98(n, m, p):
    """Rank-98 decomposition using Kronecker product of Strassen levels."""
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Test with a few random matrices using vectorized computation
    rng = np.random.default_rng(42)
    for trial in range(3):
        A = rng.integers(-3, 4, size=(n, m)).astype(float)
        B = rng.integers(-3, 4, size=(m, p)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        # Vectorized: compute all intermediate values
        U_proj = U @ A_flat  # (rank,)
        V_proj = V @ B_flat  # (rank,)
        M = U_proj * V_proj  # (rank,)
        
        # C[i,k] = sum_t W[k*n+i, t] * M[t]
        C_computed = (W * M[np.newaxis, :]).sum(axis=1).reshape(p, n).T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _reduce_rank_direct(U, V, W, n, m, p, time_budget):
    """
    Direct tensor column rank check with greedy column removal.
    """
    rank = U.shape[1]
    dim = n * m * p
    elapsed_start = time.time()
    
    # Build index arrays for tensor flattening
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_flat_idx = i_idx * m + j_idx
    v_flat_idx = j_idx * p + k_idx
    w_flat_idx = k_idx * n + i_idx
    
    U_vals = U[u_flat_idx, :]
    V_vals = V[v_flat_idx, :]
    W_vals = W[w_flat_idx, :]
    
    T = U_vals * V_vals * W_vals  # (dim, rank)
    
    if time.time() - elapsed_start > time_budget * 0.1:
        return None
    
    # Check numerical rank of T
    try:
        s = np.linalg.svd(T, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    
    tol = 1e-10
    num_rank = int(np.sum(s > tol))
    
    if num_rank >= rank:
        return None
    
    # Greedy removal: try removing columns one by one
    # Strategy: start from the full set, try removing each column
    # Prioritize columns that are linear combinations of others
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    active = list(range(rank))
    
    # Sort by singular value contribution (smallest first)
    col_norms = np.linalg.norm(T, axis=0)
    col_order = np.argsort(col_norms)
    
    removed_count = 0
    for col_to_remove in col_order:
        if time.time() - elapsed_start > time_budget * 0.7:
            break
        if len(active) <= num_rank:
            break
        if col_to_remove not in active:
            continue
        
        new_active = [c for c in active if c != col_to_remove]
        if len(new_active) < 1:
            break
        
        U_test = U[:, new_active]
        V_test = V[:, new_active]
        W_test = W[:, new_active]
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p):
            active = new_active
            U_cur = U_test
            V_cur = V_test
            W_cur = W_test
            removed_count += 1
    
    if removed_count > 0:
        return (U_cur, V_cur, W_cur)
    
    # If simple removal didn't work, try expressing dependent columns
    # as linear combinations and folding into W
    if num_rank < rank and time.time() - elapsed_start < time_budget * 0.8:
        try:
            # Use QR with pivoting to find independent set
            Q, R, piv = _qr_pivot(T, min(dim, rank))
            independent_cols = sorted(piv[:num_rank].tolist())
            dependent_cols = sorted(piv[num_rank:].tolist())
            
            T_indep = T[:, independent_cols]
            T_dep = T[:, dependent_cols]
            
            coeffs, _, _, _ = np.linalg.lstsq(T_indep, T_dep, rcond=1e-12)
            
            if np.allclose(T_indep @ coeffs, T_dep, atol=1e-8):
                # Fold dependent columns into W
                W_mod = W[:, independent_cols].copy()
                for d_idx, dep_col in enumerate(dependent_cols):
                    for j_idx, indep_col in enumerate(independent_cols):
                        c = coeffs[j_idx, d_idx]
                        if abs(c) > 1e-14:
                            W_mod[:, j_idx] += c * W[:, dep_col]
                
                U_reduced = U[:, independent_cols]
                V_reduced = V[:, independent_cols]
                
                if _verify_decomposition(U_reduced, V_reduced, W_mod, n, m, p):
                    return (U_reduced, V_reduced, W_mod)
                
                # Try folding into V instead
                V_mod = V[:, independent_cols].copy()
                W_reduced = W[:, independent_cols].copy()
                for d_idx, dep_col in enumerate(dependent_cols):
                    for j_idx, indep_col in enumerate(independent_cols):
                        c = coeffs[j_idx, d_idx]
                        if abs(c) > 1e-14:
                            V_mod[:, j_idx] += c * V[:, dep_col]
                
                if _verify_decomposition(U_reduced, V_mod, W_reduced, n, m, p):
                    return (U_reduced, V_mod, W_reduced)
        except Exception:
            pass
    
    return None


def _reduce_rank_rectangular(U, V, W, n, m, p, time_budget):
    """
    Exploit the two-block structure: try to merge columns across blocks
    by checking if second-block columns are linear combinations of first-block
    columns in the tensor space.
    """
    rank = U.shape[1]
    elapsed_start = time.time()
    
    dim = n * m * p
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_flat_idx = i_idx * m + j_idx
    v_flat_idx = j_idx * p + k_idx
    w_flat_idx = k_idx * n + i_idx
    
    U_vals = U[u_flat_idx, :]
    V_vals = V[v_flat_idx, :]
    W_vals = W[w_flat_idx, :]
    
    T = U_vals * V_vals * W_vals
    
    # Check if second block columns can be expressed via first block
    T_first = T[:, 0:49]
    T_second = T[:, 49:98]
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(T_first, T_second, rcond=1e-12)
        
        if np.allclose(T_first @ coeffs, T_second, atol=1e-6):
            # Second block is in span of first block
            # Fold into W: W_new[:, j] = W[:, j] + sum_k coeffs[k,j] * W[:, 49+k]
            W_mod = W[:, 0:49].copy()
            for j in range(49):
                for k in range(49):
                    c = coeffs[k, j]
                    if abs(c) > 1e-12:
                        W_mod[:, j] += c * W[:, 49+k]
            
            U_reduced = U[:, 0:49]
            V_reduced = V[:, 0:49]
            
            if _verify_decomposition(U_reduced, V_reduced, W_mod, n, m, p):
                return (U_reduced, V_reduced, W_mod)
            
            # Try folding into V instead
            V_mod = V[:, 0:49].copy()
            W_reduced = W[:, 0:49].copy()
            for j in range(49):
                for k in range(49):
                    c = coeffs[k, j]
                    if abs(c) > 1e-12:
                        V_mod[:, j] += c * V[:, 49+k]
            
            if _verify_decomposition(U_reduced, V_mod, W_reduced, n, m, p):
                return (U_reduced, V_mod, W_reduced)
            
            # Try folding into U instead
            U_mod = U[:, 0:49].copy()
            V_reduced2 = V[:, 0:49].copy()
            W_reduced2 = W[:, 0:49].copy()
            for j in range(49):
                for k in range(49):
                    c = coeffs[k, j]
                    if abs(c) > 1e-12:
                        U_mod[:, j] += c * U[:, 49+k]
            
            if _verify_decomposition(U_mod, V_reduced2, W_reduced2, n, m, p):
                return (U_mod, V_reduced2, W_reduced2)
    except Exception:
        pass
    
    if time.time() - elapsed_start > time_budget * 0.5:
        return None
    
    # Greedy removal from the full 98 columns
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    active = list(range(rank))
    
    col_norms = np.linalg.norm(T, axis=0)
    col_order = np.argsort(col_norms)
    
    for col in col_order:
        if time.time() - elapsed_start > time_budget * 0.95:
            break
        if col not in active:
            continue
        if len(active) <= 49:
            break
        
        new_active = [c for c in active if c != col]
        U_test = U[:, new_active]
        V_test = V[:, new_active]
        W_test = W[:, new_active]
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p):
            active = new_active
            U_cur = U_test
            V_cur = V_test
            W_cur = W_test
    
    if len(active) < rank:
        return (U_cur, V_cur, W_cur)
    
    return None


def _qr_pivot(A, num_rank):
    """QR decomposition with column pivoting to find independent columns."""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    piv = list(range(n))
    
    for k in range(min(m, n)):
        # Find column with largest norm in remaining part
        if k < n:
            norms = np.linalg.norm(R[k:, k:], axis=0)
            max_col = np.argmax(norms) + k
            if max_col != k:
                R[:, [k, max_col]] = R[:, [max_col, k]]
                piv[k], piv[max_col] = piv[max_col], piv[k]
        
        # QR step
        col_norm = np.linalg.norm(R[k:, k])
        if col_norm < 1e-14:
            break
        v = R[k:, k] / col_norm
        R[k:, k:] -= col_norm * np.outer(v, R[k:, k:] @ v)
        R[k, k] = col_norm
        Q[k:, :] -= col_norm * np.outer(v, Q[k:, :] @ v)
    
    return Q, R, piv
