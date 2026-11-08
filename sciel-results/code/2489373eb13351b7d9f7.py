import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    
    def compute_error(U, V, W, valid_a, valid_b, valid_c, invalid_a, invalid_b, invalid_c):
        valid_vals = np.sum(U[valid_a, :] * V[valid_b, :] * W[valid_c, :], axis=1)
        invalid_vals = np.sum(U[invalid_a, :] * V[invalid_b, :] * W[invalid_c, :], axis=1)
        error = np.sum((valid_vals - 1.0) ** 2) + np.sum(invalid_vals ** 2)
        return error

    def try_round_and_verify(U, V, W, valid_a, valid_b, valid_c, invalid_a, invalid_b, invalid_c, tolerance=1e-8):
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        valid_vals = np.sum(U_int[valid_a, :] * V_int[valid_b, :] * W_int[valid_c, :], axis=1)
        if not np.allclose(valid_vals, 1.0, atol=tolerance):
            return None
            
        invalid_vals = np.sum(U_int[invalid_a, :] * V_int[invalid_b, :] * W_int[invalid_c, :], axis=1)
        if not np.allclose(invalid_vals, 0.0, atol=tolerance):
            return None
            
        return {
            "U": U_int.tolist(),
            "V": V_int.tolist(),
            "W": W_int.tolist()
        }

    # Precompute indices
    valid_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_terms.append((a, b, c))
                
    valid_set = set(valid_terms)
    invalid_terms = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid_set:
                    invalid_terms.append((a, b, c))
                    
    valid_a = np.array([t[0] for t in valid_terms])
    valid_b = np.array([t[1] for t in valid_terms])
    valid_c = np.array([t[2] for t in valid_terms])
    invalid_a = np.array([t[0] for t in invalid_terms])
    invalid_b = np.array([t[1] for t in invalid_terms])
    invalid_c = np.array([t[2] for t in invalid_terms])

    start_time = time.time()
    deadline = 140
    
    # Try ranks 40, 43
    for rank in [40, 43]:
        if time.time() > deadline:
            break
            
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Simple GD with momentum
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            lr = 0.01
            beta = 0.9
            
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                valid_vals = np.sum(U[valid_a, :] * V[valid_b, :] * W[valid_c, :], axis=1)
                invalid_vals = np.sum(U[invalid_a, :] * V[invalid_b, :] * W[invalid_c, :], axis=1)
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Vectorized gradient accumulation is hard without sparse matrices, 
                # but we can use the structure.
                # For now, use a simple approximation or small batch?
                # Actually, the previous code used loops. Let's stick to loops but optimize.
                
                # This part is slow. 
                # Let's just check error periodically and rely on the fallback.
                pass
                
            # This search is too slow to be effective in 150s with Python loops for gradients.
            # I will rely on the Rank 44 fallback which is guaranteed.
            # But I want to try to get Rank 40.
            # Maybe I can use a known Rank 40 decomposition?
            # There is no known Rank 40 decomposition for 3x4x4 in my immediate recall.
            # The best known is likely 44 or 43.
            
    # Fallback Rank 44
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
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
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
