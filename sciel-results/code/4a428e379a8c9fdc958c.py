"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    import numpy as np
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 150s limit

    # 2x2x2 Strassen rank-7 decomposition
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=float)
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=float)
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=float)

    n_blocks = n // 2  # 2
    m_blocks = m // 2  # 2
    p_blocks = p // 2  # 3

    total_rank = n_blocks * m_blocks * p_blocks * 7  # 84

    # Build U, V, W matrices
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j, col] = S_U[di * 2 + dj, c]
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V[j * p + k, col] = S_V[dj * 2 + dk, c]
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W[k * n + i, col] = S_W[dk * 2 + di, c]
                col_offset += 7

    # Verify the decomposition
    def verify_decomp(U_mat, V_mat, W_mat):
        """Check if decomposition is valid"""
        rank = U_mat.shape[1]
        T_decomp = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u_col = U_mat[:, r]
            v_col = V_mat[:, r]
            w_col = W_mat[:, r]
            for i in range(n):
                for k in range(p):
                    u_vals = u_col[i*m:(i+1)*m]
                    v_vals = v_col[k::p][:m]
                    w_val = w_col[k*n+i]
                    T_decomp[i, :, k] += u_vals * v_vals * w_val
        return np.allclose(T_decomp, 1.0, atol=1e-9)

    # Verify current decomposition
    if not verify_decomp(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by removing redundant columns
    # Precompute tensor contributions for all columns
    def compute_tensor_col(U_mat, V_mat, W_mat, r):
        u_col = U_mat[:, r]
        v_col = V_mat[:, r]
        w_col = W_mat[:, r]
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for k in range(p):
                u_vals = u_col[i*m:(i+1)*m]
                v_vals = v_col[k::p][:m]
                w_val = w_col[k*n+i]
                T[i, :, k] += u_vals * v_vals * w_val
        return T

    tensor_cols = [compute_tensor_col(U, V, W, r) for r in range(total_rank)]
    
    # Flatten to vectors for SVD
    vec_dim = n * m * p
    tensor_matrix = np.zeros((total_rank, vec_dim), dtype=float)
    for r in range(total_rank):
        tensor_matrix[r] = tensor_cols[r].flatten()
    
    # Use SVD to find effective rank
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(tensor_matrix, full_matrices=False)
        tol = 1e-10
        effective_rank = np.sum(s_vals > tol)
        
        if effective_rank < total_rank:
            # Try to remove redundant columns greedily
            current_U = U.copy()
            current_V = V.copy()
            current_W = W.copy()
            current_rank = total_rank
            removed = [False] * total_rank
            
            # Sort columns by their projection onto null space
            null_space = Vt_svd[effective_rank:]
            null_proj = np.zeros(total_rank, dtype=float)
            for r in range(total_rank):
                null_proj[r] = np.linalg.norm(null_space[:, r])
            
            col_order = np.argsort(-null_proj)
            
            max_checks = 30
            checks_done = 0
            
            for r in col_order:
                if checks_done >= max_checks:
                    break
                if time.time() > deadline - 15:
                    break
                if removed[r]:
                    continue
                
                # Try removing column r
                new_U = np.delete(current_U, r, axis=1)
                new_V = np.delete(current_V, r, axis=1)
                new_W = np.delete(current_W, r, axis=1)
                
                if verify_decomp(new_U, new_V, new_W):
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    current_rank = current_rank - 1
                    removed[r] = True
                checks_done += 1
            
            # Rebuild arrays without removed columns
            if any(removed):
                keep_indices = [r for r in range(total_rank) if not removed[r]]
                current_U = U[:, keep_indices]
                current_V = V[:, keep_indices]
                current_W = W[:, keep_indices]
                current_rank = len(keep_indices)
            
            # Final verification
            if verify_decomp(current_U, current_V, current_W):
                return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
        
    except Exception as e:
        pass

    # Fallback to original
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
