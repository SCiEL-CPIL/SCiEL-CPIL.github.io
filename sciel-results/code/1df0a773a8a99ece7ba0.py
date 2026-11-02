import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
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
    
    # Try to find a lower-rank local 3x2x2 decomposition using optimization
    # Build the tensor for 3x2x2
    # T[a_idx, b_idx, c_idx] where:
    # a_idx = a_row * 2 + a_col (0..5)
    # b_idx = b_row * 2 + b_col (0..3)
    # c_idx = c_row * 2 + c_col (0..5)
    # T = 1 if c_row==a_row and there exists a_col, b_row such that 
    #     a_col == b_row and c_col == b_col, i.e., standard matrix mult
    
    T_local = np.zeros((6, 4, 6))
    for a_row in range(3):
        for a_col in range(2):
            for b_row in range(2):
                for b_col in range(2):
                    # C[a_row, b_col] += A[a_row, a_col] * B[a_col, b_col]
                    # Wait, B is 2x2, so B[b_row, b_col]
                    # C[a_row, b_col] = sum_{a_col} A[a_row, a_col] * B[a_col, b_col]
                    # So the contribution to C[a_row, b_col] from A[a_row, a_col] and B[a_col, b_col]
                    c_row = a_row
                    c_col = b_col
                    T_local[a_row*2+a_col, a_col*2+b_col, c_row*2+c_col] += 1
    
    # Try rank 10 optimization
    if time.time() < deadline - 30:
        try:
            best_rank = 11
            best_U, best_V, best_W = U_loc.copy(), V_loc.copy(), W_loc.copy()
            
            for target_rank in [10, 9]:
                if time.time() > deadline - 20:
                    break
                
                # Initialize with random factors
                np.random.seed(42)
                U_opt = np.random.randn(6, target_rank)
                V_opt = np.random.randn(4, target_rank)
                W_opt = np.random.randn(6, target_rank)
                
                # Optimize using gradient-free approach (Nelder-Mead style)
                # We want to minimize sum of squared errors
                def compute_error(params):
                    U_p = params[:6*target_rank].reshape(6, target_rank)
                    V_p = params[6*target_rank:6*target_rank+4*target_rank].reshape(4, target_rank)
                    W_p = params[6*target_rank+4*target_rank:].reshape(6, target_rank)
                    
                    # Compute reconstructed tensor
                    T_recon = np.zeros((6, 4, 6))
                    for r in range(target_rank):
                        T_recon += np.einsum('i,j,k->ijk', U_p[:,r], V_p[:,r], W_p[:,r])
                    
                    err = np.sum((T_local - T_recon)**2)
                    return err
                
                # Use scipy minimize
                from scipy.optimize import minimize
                x0 = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
                
                result = minimize(compute_error, x0, method='Nelder-Mead', 
                                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if result.fun < 1e-6:
                    best_rank = target_rank
                    best_U = result.x[:6*target_rank].reshape(6, target_rank)
                    best_V = result.x[6*target_rank:6*target_rank+4*target_rank].reshape(4, target_rank)
                    best_W = result.x[6*target_rank+4*target_rank:].reshape(6, target_rank)
                    break
            
            if best_rank < 11:
                # Use the optimized local factors
                U_loc = best_U
                V_loc = best_V
                W_loc = best_W
                rank = 4 * best_rank
                
                # Rebuild U, V, W with new rank
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                def add_block_opt(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
                    col_offset = block_idx * best_rank
                    
                    for r in range(3):
                        for c in range(2):
                            local_a_idx = r * 2 + c
                            global_a_row = a_row_start + r
                            global_a_col = a_col_start + c
                            global_a_idx = global_a_row * m + global_a_col
                            U[global_a_idx, col_offset:col_offset+best_rank] += U_loc[local_a_idx, :]
                    
                    for r in range(2):
                        for c in range(2):
                            local_b_idx = r * 2 + c
                            global_b_row = b_row_start + r
                            global_b_col = b_col_start + c
                            global_b_idx = global_b_row * p + global_b_col
                            V[global_b_idx, col_offset:col_offset+best_rank] += V_loc[local_b_idx, :]
                    
                    for r in range(3):
                        for c in range(2):
                            local_c_idx = r * 2 + c
                            global_c_row = c_row_start + r
                            global_c_col = c_col_start + c
                            global_c_idx = global_c_col * n + global_c_row
                            W[global_c_idx, col_offset:col_offset+best_rank] += W_loc[local_c_idx, :]
                
                add_block_opt(0, 0, 0, 0, 0, 0, 0)
                add_block_opt(1, 0, 2, 2, 0, 0, 0)
                add_block_opt(2, 0, 0, 0, 2, 0, 2)
                add_block_opt(3, 0, 2, 2, 2, 0, 2)
                
                # Round to half-integers if possible
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify
                U_np = np.array(U)
                V_np = np.array(V)
                W_np = np.array(W)
                
                # Build global tensor
                T_global = np.zeros((n*m, m*p, p*n))
                for a_row in range(n):
                    for a_col in range(m):
                        for b_row in range(m):
                            for b_col in range(p):
                                c_row = a_row
                                c_col = b_col
                                T_global[a_row*m+a_col, b_row*p+b_col, c_col*n+c_row] += 1
                
                T_recon = np.zeros((n*m, m*p, p*n))
                for r in range(rank):
                    T_recon += np.einsum('i,j,k->ijk', U_np[:,r], V_np[:,r], W_np[:,r])
                
                if np.allclose(T_global, T_recon, atol=1e-6):
                    return {
                        "U": U_np.tolist(),
                        "V": V_np.tolist(),
                        "W": W_np.tolist()
                    }
        except Exception:
            pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
