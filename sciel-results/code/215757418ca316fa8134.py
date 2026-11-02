import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify tensor decomposition correctness."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for a in range(n * m):
            a_row, a_col = divmod(a, m)
            for b in range(m * p):
                b_row, b_col = divmod(b, p)
                for c in range(p * n):
                    c_col, c_row = divmod(c, n)
                    expected = 1.0 if (a_col == b_row and c_row == a_row and c_col == b_col) else 0.0
                    computed = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(computed - expected) > tol:
                        return False
        return True

    def _build_base_decomposition():
        """Build the rank-44 base decomposition using Strassen for rows 0-1 + trivial for row 2."""
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Local factors for 3x2x2 multiplication
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen part (cols 0-6) for 2x2x2
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Extra part (cols 7-10) for Row 2 of A
        U_loc[4, 7] = 1
        V_loc[0, 7] = 1
        W_loc[4, 7] = 1
        
        U_loc[5, 8] = 1
        V_loc[2, 8] = 1
        W_loc[4, 8] = 1
        
        U_loc[4, 9] = 1
        V_loc[1, 9] = 1
        W_loc[5, 9] = 1
        
        U_loc[5, 10] = 1
        V_loc[3, 10] = 1
        W_loc[5, 10] = 1
        
        # Verify local decomposition
        for a in range(6):
            for b in range(4):
                for c in range(6):
                    a_row, a_col = divmod(a, 2)
                    b_row, b_col = divmod(b, 2)
                    c_row, c_col = divmod(c, 2)
                    expected = 1.0 if (a_col == b_row and c_row == a_row and c_col == b_col) else 0.0
                    computed = np.sum(U_loc[a, :] * V_loc[b, :] * W_loc[c, :])
                    if abs(computed - expected) > 1e-10:
                        raise ValueError(f"Local decomposition error at ({a},{b},{c}): {computed} != {expected}")
        
        def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_row = a_row_start + r
                    global_a_col = a_col_start + c
                    global_a_idx = global_a_row * m + global_a_col
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_row = b_row_start + r
                    global_b_col = b_col_start + c
                    global_b_idx = global_b_row * p + global_b_col
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_row = c_row_start + r
                    global_c_col = c_col_start + c
                    global_c_idx = global_c_col * n + global_c_row
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
        
        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        return U, V, W

    def _try_reduce_rank(U, V, W, max_time=60):
        """Try to reduce rank by removing redundant columns via linear dependency analysis."""
        start = time.time()
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = rank
        
        # Try removing one column at a time
        for col in range(rank):
            if time.time() - start > max_time:
                break
            U_test = np.delete(U_arr, col, axis=1)
            V_test = np.delete(V_arr, col, axis=1)
            W_test = np.delete(W_arr, col, axis=1)
            
            if _verify_decomposition(U_test, V_test, W_test, n, m, p, tol=1e-10):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = rank - 1
                rank -= 1
                # Reset U_arr etc. to the new best
                U_arr = best_U
                V_arr = best_V
                W_arr = best_W
                # Restart the loop from beginning since indices changed
                col = -1
        
        return best_U, best_V, best_W

    def _try_combined_blocks():
        """
        Try to find a lower-rank decomposition by combining the 4 blocks.
        Key idea: the 4 trivial columns for row 2 in each block might be shareable
        if we reformulate. Also try rank-10 for 3x2x2 by sharing Strassen products
        with row-2 products.
        """
        # First, try to find rank-10 for 3x2x2
        # The 2x2x2 part needs rank 7 (Strassen). Row 2 needs:
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        # This is a 1x2 * 2x2 -> 1x2 multiplication, which has rank 4 (trivial) 
        # but can we do it in 3? The tensor for 1x2*2x2->1x2 has rank 4.
        # So 7+4=11 is likely minimal for 3x2x2 with this tiling.
        
        # Alternative: use a different tiling. What if we tile as 2x4 * 4x2 -> 2x2 blocks?
        # A is 3x4, B is 4x4, C is 3x4
        # Tile A as (2x2 + 1x2) rows, B as 2x2 cols, C as (2x2 + 1x2) rows
        # Block structure:
        #   A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]  (2x2*2x2, rank 7)
        #   A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]  (2x2*2x2, rank 7)
        #   A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]  (2x2*2x2, rank 7)
        #   A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]  (2x2*2x2, rank 7)
        #   A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]  (1x2*2x2, rank 4)
        #   A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]  (1x2*2x2, rank 4)
        #   A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]  (1x2*2x2, rank 4)
        #   A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]  (1x2*2x2, rank 4)
        # Total: 4*7 + 4*4 = 28 + 16 = 44. Same as before.
        
        # What about using a 3x3*3x3 block? No, dimensions don't work out.
        
        # Try: can we share products between the 4 rank-7 Strassen blocks?
        # The Strassen products for different blocks use different A and B submatrices,
        # so direct sharing isn't possible. But maybe we can combine some.
        
        # For now, just return the base decomposition and let _try_reduce_rank work on it
        return None

    # Build base decomposition
    U, V, W = _build_base_decomposition()
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to trivial
        rank_trivial = n * m * p  # 48
        U_f = np.zeros((n*m, rank_trivial))
        V_f = np.zeros((m*p, rank_trivial))
        W_f = np.zeros((p*n, rank_trivial))
        for a_idx in range(n*m):
            a_r, a_c = divmod(a_idx, m)
            for b_idx in range(m*p):
                b_r, b_c = divmod(b_idx, p)
                if a_c == b_r:
                    c_idx = b_c * n + a_r
                    rank_idx = a_idx * (m*p) + b_idx
                    U_f[a_idx, rank_idx] = 1
                    V_f[b_idx, rank_idx] = 1
                    W_f[c_idx, rank_idx] = 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    # Try to reduce rank
    if time.time() < deadline - 10:
        U, V, W = _try_reduce_rank(U, V, W, max_time=50)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to trivial
        rank_trivial = n * m * p
        U_f = np.zeros((n*m, rank_trivial))
        V_f = np.zeros((m*p, rank_trivial))
        W_f = np.zeros((p*n, rank_trivial))
        for a_idx in range(n*m):
            a_r, a_c = divmod(a_idx, m)
            for b_idx in range(m*p):
                b_r, b_c = divmod(b_idx, p)
                if a_c == b_r:
                    c_idx = b_c * n + a_r
                    rank_idx = a_idx * (m*p) + b_idx
                    U_f[a_idx, rank_idx] = 1
                    V_f[b_idx, rank_idx] = 1
                    W_f[c_idx, rank_idx] = 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
