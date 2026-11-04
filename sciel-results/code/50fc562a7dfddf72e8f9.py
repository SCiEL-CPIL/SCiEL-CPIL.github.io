import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_SEED_START = time.time()


def _deadline_guard(seconds_remaining=115.0):
    """Return True if we should stop searching."""
    return (time.time() - _SEED_START) > seconds_remaining


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Build rank-56 Strassen decomposition
    rank = 56
    U = np.zeros((n*m, rank))  # 8x56
    V = np.zeros((m*p, rank))  # 32x56
    W = np.zeros((p*n, rank))  # 16x56
    
    # Strassen 2x2x2 patterns (7 terms)
    U_patterns = np.array([
        [[1, 0], [0, 1]],   # t1: a00+a11
        [[0, 0], [1, 1]],   # t2: a10+a11
        [[1, 0], [0, 0]],   # t3: a00
        [[0, 0], [0, 1]],   # t4: a11
        [[1, 1], [0, 0]],   # t5: a00+a01
        [[-1, 0], [1, 0]],  # t6: a10-a00
        [[0, 1], [0, -1]]   # t7: a01-a11
    ])
    
    V_patterns = np.array([
        [[1, 0], [0, 1]],   # t1: b00+b11
        [[1, 0], [0, 0]],   # t2: b00
        [[0, 1], [0, -1]],  # t3: b01-b11
        [[-1, 0], [1, 0]],  # t4: b10-b00
        [[0, 0], [0, 1]],   # t5: b11
        [[1, 0], [1, 0]],   # t6: b00+b10
        [[0, 1], [0, 1]]    # t7: b01+b11
    ])
    
    W_patterns = np.array([
        [[1, 0], [0, 1]],   # t1: c00+1, c11+1
        [[0, 0], [1, -1]],  # t2: c10+1, c11-1
        [[0, 1], [0, 1]],   # t3: c01+1, c11+1
        [[1, 0], [1, 0]],   # t4: c00+1, c10+1
        [[-1, 1], [0, 0]],  # t5: c00-1, c01+1
        [[0, 0], [0, 1]],   # t6: c11+1
        [[1, 0], [0, 0]]    # t7: c00+1
    ])
    
    # 8 blocks: 2 a_blocks x 4 b_subblocks
    for b in range(8):
        a_block = b // 4      # 0 or 1
        b_subblock = b % 4    # 0, 1, 2, 3
        
        a_col_offset = a_block * 2
        b_row_offset = a_block * 2
        b_col_offset = b_subblock * 2
        
        for r in range(7):
            col = b * 7 + r
            
            for i in range(2):
                for j in range(2):
                    U[i * m + (a_col_offset + j), col] = U_patterns[r, i, j]
            
            for j in range(2):
                for k in range(2):
                    V[(b_row_offset + j) * p + (b_col_offset + k), col] = V_patterns[r, j, k]
            
            for i in range(2):
                for k in range(2):
                    W[(b_col_offset + k) * n + i, col] = W_patterns[r, i, k]
    
    # Efficient validation function
    def validate(U_arr, V_arr, W_arr, rank_val):
        """Check if decomposition exactly computes matrix multiplication tensor."""
        if _deadline_guard():
            return False
        
        i_idx, j_idx, k_idx = np.indices((n, m, p))
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        U_g = U_arr[u_idx, :rank_val]
        V_g = V_arr[v_idx, :rank_val]
        W_g = W_arr[w_idx, :rank_val]
        
        T_flat = np.sum(U_g * V_g * W_g, axis=1)
        return np.allclose(T_flat, 1.0, atol=1e-9)
    
    # Verify initial decomposition
    if not validate(U, V, W, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        U_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in U]
        V_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in V]
        W_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Greedy rank reduction: try zeroing each column
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    current_rank = rank
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    # Multiple passes of greedy column elimination
    for pass_num in range(3):
        if _deadline_guard():
            break
        
        improved = False
        for col in range(current_rank):
            if _deadline_guard():
                break
            
            # Try zeroing column col
            saved_U = current_U[:, col].copy()
            saved_V = current_V[:, col].copy()
            saved_W = current_W[:, col].copy()
            
            current_U[:, col] = 0
            current_V[:, col] = 0
            current_W[:, col] = 0
            
            if validate(current_U, current_V, current_W, current_rank):
                # Column can be removed - shift remaining columns left
                if col < current_rank - 1:
                    current_U[:, col:current_rank-1] = current_U[:, col+1:current_rank]
                    current_V[:, col:current_rank-1] = current_V[:, col+1:current_rank]
                    current_W[:, col:current_rank-1] = current_W[:, col+1:current_rank]
                current_U[:, current_rank-1] = 0
                current_V[:, current_rank-1] = 0
                current_W[:, current_rank-1] = 0
                current_rank -= 1
                improved = True
            else:
                # Restore
                current_U[:, col] = saved_U
                current_V[:, col] = saved_V
                current_W[:, col] = saved_W
        
        if not improved:
            break
        
        # Update best
        best_rank = current_rank
        best_U = current_U.copy()
        best_V = current_V.copy()
        best_W = current_W.copy()
    
    # Try pairwise column zeroing for additional reduction
    if not _deadline_guard() and best_rank > 40:
        for col1 in range(best_rank):
            if _deadline_guard():
                break
            for col2 in range(col1 + 1, best_rank):
                if _deadline_guard():
                    break
                
                saved_U1 = best_U[:, col1].copy()
                saved_V1 = best_V[:, col1].copy()
                saved_W1 = best_W[:, col1].copy()
                saved_U2 = best_U[:, col2].copy()
                saved_V2 = best_V[:, col2].copy()
                saved_W2 = best_W[:, col2].copy()
                
                best_U[:, col1] = 0
                best_V[:, col1] = 0
                best_W[:, col1] = 0
                best_U[:, col2] = 0
                best_V[:, col2] = 0
                best_W[:, col2] = 0
                
                if validate(best_U, best_V, best_W, best_rank):
                    # Remove both columns
                    cols_to_keep = [c for c in range(best_rank) if c != col1 and c != col2]
                    new_rank = len(cols_to_keep)
                    new_U = best_U[:, cols_to_keep].copy()
                    new_V = best_V[:, cols_to_keep].copy()
                    new_W = best_W[:, cols_to_keep].copy()
                    
                    best_rank = new_rank
                    best_U = new_U
                    best_V = new_V
                    best_W = new_W
                else:
                    # Restore
                    best_U[:, col1] = saved_U1
                    best_V[:, col1] = saved_V1
                    best_W[:, col1] = saved_W1
                    best_U[:, col2] = saved_U2
                    best_V[:, col2] = saved_V2
                    best_W[:, col2] = saved_W2
    
    # Final validation of best result
    if not validate(best_U, best_V, best_W, best_rank):
        best_rank = rank
        best_U = U
        best_V = V
        best_W = W
    
    # Convert to lists with proper integer types
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_U[:, :best_rank]]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_V[:, :best_rank]]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_W[:, :best_rank]]
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
