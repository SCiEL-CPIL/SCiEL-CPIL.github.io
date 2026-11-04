import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen 2x2*2x2 factors
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Part 1: 4x4*4x4 Strassen (rank 49 = 7*7)
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2 (rank 28 = 4*7)
    block_products = [
        (0, 0, 0, 4, 0, 4),
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    def _verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if decomposition is exact."""
        n, m, p = 4, 4, 6
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = np.dot(U_mat[u, :] * V_mat[v, :], W_mat[w, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Verify base decomposition
    base_err = _verify(U, V, W)
    
    # Try to compress by identifying linearly dependent columns
    # Strategy: For each column, check if it can be expressed as a linear combination
    # of other columns while preserving the tensor decomposition property
    
    U_orig = U.copy()
    V_orig = V.copy()
    W_orig = W.copy()
    
    # Build the tensor entries list
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_entries.append((i * m + j, j * p + k, k * n + i))
    
    def check_error(U_mat, V_mat, W_mat):
        mx = 0.0
        for (ui, vi, wi) in tensor_entries:
            val = np.dot(U_mat[ui, :] * V_mat[vi, :], W_mat[wi, :])
            e = abs(val - 1.0)
            if e > mx:
                mx = e
        return mx
    
    # Try greedy column elimination: remove columns that are linearly dependent
    # and can be compensated by adjusting remaining columns
    current_U = U_orig.copy()
    current_V = V_orig.copy()
    current_W = W_orig.copy()
    current_rank = total_rank
    removed = set()
    
    # First pass: try removing each column and see if the remaining can still
    # span the same tensor (using least squares to find corrections)
    for trial in range(total_rank):
        if time.time() > deadline - 10:
            break
        
        col_idx = trial
        if col_idx in removed:
            continue
            
        # Temporarily zero this column
        current_U[:, col_idx] = 0
        current_V[:, col_idx] = 0
        current_W[:, col_idx] = 0
        
        # Check if error is still small (meaning this column was redundant)
        err = check_error(current_U, current_V, current_W)
        if err < 1e-9:
            removed.add(col_idx)
        else:
            # Restore
            current_U[:, col_idx] = U_orig[:, col_idx]
            current_V[:, col_idx] = V_orig[:, col_idx]
            current_W[:, col_idx] = W_orig[:, col_idx]
    
    # Compact the remaining columns
    keep_cols = [c for c in range(total_rank) if c not in removed]
    final_rank = len(keep_cols)
    
    if final_rank < total_rank:
        final_U = current_U[:, keep_cols]
        final_V = current_V[:, keep_cols]
        final_W = current_W[:, keep_cols]
    else:
        final_U = U_orig
        final_V = V_orig
        final_W = W_orig
    
    # Round to integers/half-integers
    final_U = np.round(final_U * 2) / 2
    final_V = np.round(final_V * 2) / 2
    final_W = np.round(final_W * 2) / 2
    
    # Final verification
    final_err = check_error(final_U, final_V, final_W)
    
    if final_err > 1e-8:
        # Fallback to original if compression broke it
        final_U = U_orig
        final_V = V_orig
        final_W = W_orig
    
    def _clean_matrix(mat):
        out = []
        for row in mat:
            cleaned = []
            for entry in row:
                # Try to represent as integer or half-integer
                half = round(entry * 2) / 2
                if abs(entry - half) < 1e-9:
                    if half == int(half):
                        cleaned.append(int(half))
                    else:
                        cleaned.append(float(half))
                else:
                    cleaned.append(float(entry))
            out.append(cleaned)
        return out

    return {"U": _clean_matrix(final_U), "V": _clean_matrix(final_V), "W": _clean_matrix(final_W)}
