import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 4, 4
    
    # Local tensor for M_{3,2,2}
    T_local = np.zeros((6, 4, 6))
    for a_row in range(3):
        for a_col in range(2):
            for b_col in range(2):
                a_idx = a_row * 2 + a_col
                b_idx = a_col * 2 + b_col
                c_idx = a_row * 2 + b_col
                T_local[a_idx, b_idx, c_idx] = 1.0
    
    # Try to find rank-7 decomposition for M_{3,2,2}
    found_rank7 = False
    best_U7, best_V7, best_W7 = None, None, None
    
    if time.time() < deadline - 40:
        np.random.seed(12345)
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            U7 = np.random.randn(6, 7)
            V7 = np.random.randn(4, 7)
            W7 = np.random.randn(6, 7)
            
            for it in range(150):
                if time.time() > deadline - 10:
                    break
                # Update W
                for c_idx in range(6):
                    T_flat = T_local[:, :, c_idx].flatten()
                    UV_mat = np.einsum('ir,jr->rij', U7, V7).reshape(7, 24)
                    W7[c_idx, :] = np.linalg.lstsq(UV_mat.T, T_flat, rcond=None)[0]
                # Update U
                for a_idx in range(6):
                    T_flat = T_local[a_idx, :, :].flatten()
                    VW_mat = np.einsum('jr,kr->rjk', V7, W7).reshape(7, 24)
                    U7[a_idx, :] = np.linalg.lstsq(VW_mat.T, T_flat, rcond=None)[0]
                # Update V
                for b_idx in range(4):
                    T_flat = T_local[:, b_idx, :].flatten()
                    UW_mat = np.einsum('ir,kr->rik', U7, W7).reshape(7, 36)
                    V7[b_idx, :] = np.linalg.lstsq(UW_mat.T, T_flat, rcond=None)[0]
            
            computed = np.einsum('ir,jr,kr->ijk', U7, V7, W7)
            error = np.max(np.abs(computed - T_local))
            if error < 1e-8:
                found_rank7 = True
                best_U7, best_V7, best_W7 = U7.copy(), V7.copy(), W7.copy()
                break
    
    if found_rank7:
        rank = 28
        U = np.zeros((12, rank))
        V = np.zeros((16, rank))
        W = np.zeros((12, rank))
        
        blocks = [
            (0, 0, 0, 0, 0, 0),
            (0, 2, 2, 0, 0, 0),
            (0, 0, 0, 2, 0, 2),
            (0, 2, 2, 2, 0, 2),
        ]
        
        for block_idx, (a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start) in enumerate(blocks):
            col_offset = block_idx * 7
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                    U[global_a_idx, col_offset:col_offset+7] = best_U7[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                    V[global_b_idx, col_offset:col_offset+7] = best_V7[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                    W[global_c_idx, col_offset:col_offset+7] = best_W7[local_c_idx, :]
        
        T = np.zeros((12, 16, 12))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    T[i, j, k] = 1.0
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        error = np.max(np.abs(computed - T))
        if error > 1e-8:
            found_rank7 = False
    
    if not found_rank7:
        # Fall back to rank-32 decomposition
        local_U = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
        ], dtype=float)
        local_V = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
        ], dtype=float)
        local_W = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ], dtype=float)
        
        rank = 32
        U = np.zeros((12, rank))
        V = np.zeros((16, rank))
        W = np.zeros((12, rank))
        
        blocks = [
            (0, 0, 0, 0, 0, 0),
            (0, 2, 2, 0, 0, 0),
            (0, 0, 0, 2, 0, 2),
            (0, 2, 2, 2, 0, 2),
        ]
        
        for block_idx, (a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start) in enumerate(blocks):
            col_offset = block_idx * 8
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                    U[global_a_idx, col_offset:col_offset+8] = local_U[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                    V[global_b_idx, col_offset:col_offset+8] = local_V[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                    W[global_c_idx, col_offset:col_offset+8] = local_W[local_c_idx, :]
        
        T = np.zeros((12, 16, 12))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    T[i, j, k] = 1.0
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        error = np.max(np.abs(computed - T))
        
        if error > 1e-9:
            U = np.zeros((12, 48))
            V = np.zeros((16, 48))
            W = np.zeros((12, 48))
            for row_a in range(3):
                for col_a in range(4):
                    for col_c in range(4):
                        idx = (row_a * 4 + col_a) * 4 + col_c
                        i = row_a * 4 + col_a
                        j = col_a * 4 + col_c
                        k = col_c * 3 + row_a
                        U[i, idx] = 1
                        V[j, idx] = 1
                        W[k, idx] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
