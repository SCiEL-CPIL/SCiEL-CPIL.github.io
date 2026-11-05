"""Combined program: Strassen-based structural decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = n * m * p

    def build_strassen_decomposition():
        """
        Build a Strassen-based decomposition for 4x4x7 matrix multiplication.
        
        Strategy:
        - Decompose 4x4x7 as 4x4x4 (Strassen rank 49) + 4x4x2 (Block Strassen rank 28) + 4x4x1 (Schoolbook rank 16)
        - Total rank: 93
        """
        total_rank = 93
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        current_rank = 0
        
        # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub)
        # x_sub, y_sub are 2x2 dicts, c_sub is 2x2 dict for output
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        ]
        
        # Strassen 4x4x4: 7 outer terms, each with 7 inner terms
        strassen_4x4_outer = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}),
            ({0: 1}, {1: 1, 3: -1}),
            ({3: 1}, {2: 1, 0: -1}),
            ({0: 1, 1: 1}, {3: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}),
        ]
        
        # For 4x4x4, the c_block for each outer term m_idx
        c_blocks_4x4 = [
            {(0,0): 1.0, (1,1): 1.0},
            {(1,0): 1.0, (1,1): -1.0},
            {(0,1): 1.0, (1,1): 1.0},
            {(0,0): 1.0, (1,0): 1.0},
            {(0,0): -1.0, (0,1): 1.0},
            {(1,1): 1.0},
            {(0,0): 1.0},
        ]
        
        # Part 1: 4x4x4 Strassen (rank 49)
        for m_idx in range(7):
            x_def, y_def = strassen_4x4_outer[m_idx]
            c_block = c_blocks_4x4[m_idx]
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                # Compute U coefficients
                for (li, lj), cx in x_sub.items():
                    for b, cb in x_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        U[i * m + j, r] += cx * cb
                
                # Compute V coefficients
                for (li, lj), cy in y_sub.items():
                    for b, cb in y_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        V[i * p + j, r] += cy * cb
                
                # Compute W coefficients
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    for (br, bc), cb_c in c_block.items():
                        if cb_c == 0:
                            continue
                        i_final = 2 * br + li
                        k_final = 2 * bc + lj
                        W[k_final * n + i_final, r] += cc * cb_c
        
        # Part 2: 4x4x2 Block Strassen (rank 28)
        # B columns 4, 5.
        # C = A @ B[:, 4:6]
        # Split A into 2x2 blocks A00, A01, A10, A11
        # Split B[:, 4:6] into B_top (2x2), B_bot (2x2)
        # C_top = A00 @ B_top + A01 @ B_bot
        # C_bot = A10 @ B_top + A11 @ B_bot
        # Each product is 2x2x2, rank 7.
        # 4 products total rank 28.
        
        def add_strassen_2x2(A_block_rows, A_block_cols, B_block_rows, B_block_cols, C_block_rows, C_block_cols):
            """
            Add a Strassen 2x2x2 decomposition for A_block @ B_block to the global U, V, W.
            A_block is 2x2, B_block is 2x2, C_block is 2x2.
            """
            nonlocal current_rank
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                # U: A_block entries
                for (li, lj), cx in x_sub.items():
                    i = A_block_rows[li]
                    j = A_block_cols[lj]
                    U[i * m + j, r] += cx
                    
                # V: B_block entries
                for (li, lj), cy in y_sub.items():
                    i = B_block_rows[li]
                    j = B_block_cols[lj]
                    V[i * p + j, r] += cy
                    
                # W: C_block entries
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    i_final = C_block_rows[li]
                    k_final = C_block_cols[lj]
                    W[k_final * n + i_final, r] += cc

        # Define blocks
        A00_r = [0, 1]; A00_c = [0, 1]
        A01_r = [0, 1]; A01_c = [2, 3]
        A10_r = [2, 3]; A10_c = [0, 1]
        A11_r = [2, 3]; A11_c = [2, 3]
        
        B_top_r = [0, 1]; B_top_c = [4, 5]
        B_bot_r = [2, 3]; B_bot_c = [4, 5]
        
        C_top_r = [0, 1]; C_top_c = [4, 5]
        C_bot_r = [2, 3]; C_bot_c = [4, 5]
        
        # P1 = A00 @ B_top -> C_top
        add_strassen_2x2(A00_r, A00_c, B_top_r, B_top_c, C_top_r, C_top_c)
        # P2 = A01 @ B_bot -> C_top
        add_strassen_2x2(A01_r, A01_c, B_bot_r, B_bot_c, C_top_r, C_top_c)
        # P3 = A10 @ B_top -> C_bot
        add_strassen_2x2(A10_r, A10_c, B_top_r, B_top_c, C_bot_r, C_bot_c)
        # P4 = A11 @ B_bot -> C_bot
        add_strassen_2x2(A11_r, A11_c, B_bot_r, B_bot_c, C_bot_r, C_bot_c)

        # Part 3: 4x4x1 schoolbook (rank 16)
        # B column 6.
        # C[i][6] = sum_j A[i][j] * B[j][6]
        for i in range(n):
            for j in range(m):
                r = current_rank
                current_rank += 1
                U[i * m + j, r] = 1.0
                V[j * p + 6, r] = 1.0
                W[6 * n + i, r] = 1.0
        
        assert current_rank == total_rank, f"Expected rank {total_rank}, got {current_rank}"
        
        return U.tolist(), V.tolist(), W.tolist(), total_rank

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Precompute index arrays for efficient verification
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    # Try the structural decomposition
    try:
        U_struct, V_struct, W_struct, rank_struct = build_strassen_decomposition()
        err = verify(U_struct, V_struct, W_struct, rank_struct)
        if err < 1e-8:
            best_rank = rank_struct
            best_result = {
                "U": U_struct,
                "V": V_struct,
                "W": W_struct
            }
    except Exception as e:
        pass

    # If we have a good result, try to refine it using a systematic rank reduction
    if best_rank < n * m * p and time.time() < deadline - 20:
        Ua = np.array(best_result["U"], dtype=float)
        Va = np.array(best_result["V"], dtype=float)
        Wa = np.array(best_result["W"], dtype=float)
        r = best_rank
        
        # Precompute index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        def compute_error(U, V, W):
            u_vals = U[u_idx, :]
            v_vals = V[v_idx, :]
            w_vals = W[w_idx, :]
            prod = u_vals * v_vals * w_vals
            result = prod.sum(axis=1)
            return np.max(np.abs(result - 1.0))
        
        # Try to reduce rank by finding and removing linearly dependent columns
        # A column is redundant if the tensor equation still holds without it
        # after adjusting other columns appropriately
        
        # Strategy: iteratively try to remove each column by solving for
        # compensating adjustments in the remaining columns
        current_U = Ua.copy()
        current_V = Va.copy()
        current_W = Wa.copy()
        current_r = current_U.shape[1]
        
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            if current_r <= 1:
                break
            
            # Try removing each column
            removed = False
            for col in range(current_r):
                if time.time() > deadline - 5:
                    break
                
                # Temporarily remove column
                mask = np.ones(current_r, dtype=bool)
                mask[col] = False
                U_no = current_U[:, mask]
                V_no = current_V[:, mask]
                W_no = current_W[:, mask]
                
                # Compute the residual tensor without this column
                u_vals = U_no[u_idx, :]
                v_vals = V_no[v_idx, :]
                w_vals = W_no[w_idx, :]
                prod = u_vals * v_vals * w_vals
                result_no = prod.sum(axis=1)
                
                # The missing contribution from column col
                u_col = current_U[u_idx, col]
                v_col = current_V[v_idx, col]
                w_col = current_W[w_idx, col]
                missing = u_col * v_col * w_col
                
                # We need to find adjustments to U_no, V_no, W_no such that
                # sum over remaining columns gives 1.0
                # This is underdetermined, so we use a greedy approach:
                # try to absorb the missing contribution into one other column
                
                # For each other column, try to adjust it to absorb the missing part
                # This is complex, so use a simpler heuristic:
                # Check if the missing contribution is small enough that
                # rounding errors could account for it
                max_missing = np.max(np.abs(missing))
                if max_missing < 1e-10:
                    # Column contributes almost nothing, remove it
                    current_U = U_no
                    current_V = V_no
                    current_W = W_no
                    current_r = U_no.shape[1]
                    removed = True
                    break
            
            if not removed:
                break
        
        if current_r < best_rank:
            # Round to nearest 0.5
            current_U = np.round(current_U * 2) / 2
            current_V = np.round(current_V * 2) / 2
            current_W = np.round(current_W * 2) / 2
            
            err = compute_error(current_U, current_V, current_W)
            if err < 1e-8:
                # Remove zero columns
                mask = np.any(current_U != 0, axis=0) & np.any(current_V != 0, axis=0) & np.any(current_W != 0, axis=0)
                if np.sum(mask) < current_r:
                    current_U = current_U[:, mask]
                    current_V = current_V[:, mask]
                    current_W = current_W[:, mask]
                    current_r = np.sum(mask)
                
                if current_r < best_rank:
                    best_rank = current_r
                    best_result = {
                        "U": current_U.tolist(),
                        "V": current_V.tolist(),
                        "W": current_W.tolist()
                    }
        
        # Final verification and cleanup
        Ua = np.array(best_result["U"], dtype=float)
        Va = np.array(best_result["V"], dtype=float)
        Wa = np.array(best_result["W"], dtype=float)
        Ua = np.round(Ua * 2) / 2
        Va = np.round(Va * 2) / 2
        Wa = np.round(Wa * 2) / 2
        
        err = compute_error(Ua, Va, Wa)
        if err < 1e-8:
            mask = np.any(Ua != 0, axis=0) & np.any(Va != 0, axis=0) & np.any(Wa != 0, axis=0)
            if np.sum(mask) < best_rank:
                Ua = Ua[:, mask]
                Va = Va[:, mask]
                Wa = Wa[:, mask]
                best_rank = np.sum(mask)
                best_result = {
                    "U": Ua.tolist(),
                    "V": Va.tolist(),
                    "W": Wa.tolist()
                }

    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
