"""Matrix multiplication 3x4x8 with rank-reduced decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _build_decomposition(tiling):
    """Build a decomposition given a tiling specification."""
    n, m, p = 3, 4, 8
    total_rank = 0
    blocks = []
    
    for (btype, js, je, ks, ke) in tiling:
        if btype == 'strassen2':
            r = 7
        elif btype == 'school1':
            r = 4
        else:
            raise ValueError("Unknown block type")
        blocks.append((btype, js, je, ks, ke, r))
        total_rank += r
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    col_offset = 0
    for (btype, js, je, ks, ke, r) in blocks:
        j0, j1 = js, js + 1
        k0, k1 = ks, ks + 1
        
        if btype == 'strassen2':
            u_rows = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
            for i in range(4):
                U[u_rows[i], col_offset:col_offset+7] += np.array(U2[i])
                V[v_rows[i], col_offset:col_offset+7] += np.array(V2[i])
                W[w_rows[i], col_offset:col_offset+7] += np.array(W2[i])
            col_offset += 7
        elif btype == 'school1':
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [j0 * p + k0, j1 * p + k0, j0 * p + k1, j1 * p + k1]
            w_rows = [k0 * n + 2, k1 * n + 2]
            for i in range(2):
                U[u_rows[i], col_offset:col_offset+4] += np.array(U1[i])
                W[w_rows[i], col_offset:col_offset+4] += np.array(W1[i])
            for i in range(4):
                V[v_rows[i], col_offset:col_offset+4] += np.array(V1[i])
            col_offset += 4
    
    return U, V, W


def _default_tiling():
    """Default tiling: 8 Strassen blocks for i=0,1 and 8 schoolbook for i=2."""
    tiling = []
    for jp in range(2):
        for kp in range(4):
            tiling.append(('strassen2', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    for jp in range(2):
        for kp in range(4):
            tiling.append(('school1', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    return tiling


def _compute_phi(U, V, W, n, m, p):
    """Compute the contribution matrix phi."""
    rank = U.shape[1]
    num_eqs = n * m * p
    phi = np.zeros((num_eqs, rank), dtype=float)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                phi[eq_idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                eq_idx += 1
    return phi


def _reduce_rank_fast(U, V, W, n, m, p, deadline=None):
    """
    Attempt to reduce rank by finding linearly dependent columns.
    Uses vectorized operations for speed.
    """
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    phi = _compute_phi(U, V, W, n, m, p)
    
    # Try removing each column one at a time, in order of smallest norm
    col_norms = np.linalg.norm(phi, axis=0)
    order = np.argsort(col_norms)
    
    for col_to_remove in order:
        if deadline is not None and time.time() > deadline:
            break
        
        cols_to_keep = [c for c in range(rank) if c != col_to_remove]
        A = phi[:, cols_to_keep]
        b = phi[:, col_to_remove]
        
        try:
            alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        if np.allclose(A @ alpha, b, atol=1e-9):
            new_rank = rank - 1
            if new_rank < best_rank:
                new_cols = cols_to_keep
                best_U = U[:, new_cols]
                best_V = V[:, new_cols]
                best_W = W[:, new_cols]
                best_rank = new_rank
                # Update phi and rank for subsequent iterations
                phi = phi[:, new_cols]
                rank = new_rank
                # Recompute norms
                col_norms = np.linalg.norm(phi, axis=0)
                order = np.argsort(col_norms)
    
    return best_U, best_V, best_W, best_rank


def _try_pairwise_removal(U, V, W, n, m, p, deadline=None):
    """Try removing pairs of columns if they are linearly dependent."""
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    phi = _compute_phi(U, V, W, n, m, p)
    
    # Try removing pairs
    for c1 in range(rank):
        if deadline is not None and time.time() > deadline:
            break
        for c2 in range(c1 + 1, rank):
            if deadline is not None and time.time() > deadline:
                break
            
            cols_to_keep = [c for c in range(rank) if c != c1 and c != c2]
            if len(cols_to_keep) == 0:
                continue
                
            A = phi[:, cols_to_keep]
            b1 = phi[:, c1]
            b2 = phi[:, c2]
            
            # Check if both b1 and b2 are in the column space of A
            try:
                alpha1, _, _, _ = np.linalg.lstsq(A, b1, rcond=None)
                alpha2, _, _, _ = np.linalg.lstsq(A, b2, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            if np.allclose(A @ alpha1, b1, atol=1e-9) and np.allclose(A @ alpha2, b2, atol=1e-9):
                new_rank = rank - 2
                if new_rank < best_rank:
                    best_U = U[:, cols_to_keep]
                    best_V = V[:, cols_to_keep]
                    best_W = W[:, cols_to_keep]
                    best_rank = new_rank
                    phi = phi[:, cols_to_keep]
                    rank = new_rank
    
    return best_U, best_V, best_W, best_rank


def _construct_3x4x8():
    """Construct the decomposition."""
    tiling = _default_tiling()
    U, V, W = _build_decomposition(tiling)
    
    n, m, p = 3, 4, 8
    deadline = time.time() + 100
    
    # First pass: single column removal
    U, V, W, rank1 = _reduce_rank_fast(U, V, W, n, m, p, deadline=deadline)
    
    # Second pass: try pairwise removal
    if time.time() < deadline:
        U, V, W, rank2 = _try_pairwise_removal(U, V, W, n, m, p, deadline=deadline)
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition using numpy."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True


def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    return (
        [list(row) for row in U],
        [list(row) for row in V],
        [list(row) for row in W]
    )


def construct():
    start = time.time()
    deadline = start + 110
    
    try:
        U, V, W = _construct_3x4x8()
    except Exception:
        U = V = W = None
    
    n, m, p = DIMENSIONS
    
    if U is not None:
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            U_list, V_list, W_list = _to_lists(U, V, W)
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    U_list, V_list, W_list = _to_lists(U, V, W)
    return {"U": U_list, "V": V_list, "W": W_list}
