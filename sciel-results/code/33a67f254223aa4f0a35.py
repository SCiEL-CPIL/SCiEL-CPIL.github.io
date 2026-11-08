"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # 2x2x2 Strassen rank-7 decomposition
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1],   # (1,1)
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1],   # (1,1)
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ]
    
    # Split into blocks: i:2x2, j:2x2, k:3x2
    n_blocks = n // 2  # 2
    m_blocks = m // 2  # 2
    p_blocks = p // 2  # 3
    block_size = 2
    
    total_rank = n_blocks * m_blocks * p_blocks * 7
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
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
                            U[i * m + j][col] = S_U[di * 2 + dj][c]
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V[j * p + k][col] = S_V[dj * 2 + dk][c]
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W[k * n + i][col] = S_W[dk * 2 + di][c]
                col_offset += 7
    
    # Now try to reduce rank by finding linearly dependent columns
    # The decomposition is valid if for all (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # We can check if removing a column still preserves this property.
    # A faster approach: check if any column is a linear combination of others
    # in a way that preserves the tensor.
    
    import numpy as np
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We need T[i,j,k] = 1 for all valid (i,j,k)
    
    def compute_tensor(U_arr, V_arr, W_arr):
        """Compute the tensor from the factor matrices."""
        rank = U_arr.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        return T
    
    # Verify current decomposition
    T_check = compute_tensor(U_np, V_np, W_np)
    if not np.allclose(T_check, 1.0, atol=1e-9):
        return {"U": U, "V": V, "W": W}
    
    # Try to remove columns one by one
    # For each column r, check if removing it still gives a valid decomposition
    # This is O(rank * n*m*p) per check, which is 84 * 96 = ~8000 ops per check
    # Total: 84 * 8000 = ~672,000 ops, very fast
    
    removed = set()
    for r in range(total_rank):
        if time.time() > deadline - 10:
            break
        if r in removed:
            continue
        
        # Try removing column r
        U_trial = U_np.copy()
        V_trial = V_np.copy()
        W_trial = W_np.copy()
        U_trial[:, r] = 0
        V_trial[:, r] = 0
        W_trial[:, r] = 0
        
        T_trial = compute_tensor(U_trial, V_trial, W_trial)
        if np.allclose(T_trial, 1.0, atol=1e-9):
            removed.add(r)
            U_np = U_trial
            V_np = V_trial
            W_np = W_trial
    
    # Also try removing pairs of columns if single removal didn't work
    # But first, let's see how many we removed
    if removed:
        # Reconstruct the matrices without removed columns
        keep_cols = [c for c in range(total_rank) if c not in removed]
        U_final = U_np[:, keep_cols]
        V_final = V_np[:, keep_cols]
        W_final = W_np[:, keep_cols]
        
        # Convert back to lists
        U_out = U_final.tolist()
        V_out = V_final.tolist()
        W_out = W_final.tolist()
        
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
