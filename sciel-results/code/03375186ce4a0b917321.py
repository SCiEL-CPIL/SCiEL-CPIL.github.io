import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    deadline = start_time + 110  # Safety margin

    # 1. Build the 3x3 matrix multiplication tensor
    # T[i, j, k] = 1 if A[i, j] * B[j, k] contributes to C[i, k]
    # Indices 0..8 for 3x3 matrices
    T_3x3 = np.zeros((9, 9, 9), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A index: i*3 + j
                # B index: j*3 + k
                # C index: i*3 + k
                a_idx = i * 3 + j
                b_idx = j * 3 + k
                c_idx = i * 3 + k
                T_3x3[a_idx, b_idx, c_idx] = 1.0

    # 2. Function to find a low-rank decomposition for the 3x3 tensor
    def find_decomposition(target_rank, max_iters=500, tol=1e-8, num_restarts=5):
        best_solution = None
        best_error = np.inf
        
        for restart in range(num_restarts):
            # Initialize U, V, W
            # Random initialization scaled appropriately
            U = np.random.randn(9, target_rank) * 0.5
            V = np.random.randn(9, target_rank) * 0.5
            W = np.random.randn(9, target_rank) * 0.5
            
            # Normalize initial vectors to prevent divergence
            for r in range(target_rank):
                U[:, r] /= np.linalg.norm(U[:, r]) + 1e-10
                V[:, r] /= np.linalg.norm(V[:, r]) + 1e-10
                W[:, r] /= np.linalg.norm(W[:, r]) + 1e-10
            
            current_error = np.inf
            for it in range(max_iters):
                # ALS step for U
                # T = sum U[:, r] * V[:, r] * W[:, r]
                # For fixed V, W, solve for U:
                # U[:, r] = argmin || T - sum U[:, r] * (V[:, r] * W[:, r]) ||^2
                # This is a linear least squares problem for each r?
                # No, it's coupled.
                # Standard ALS for tensor decomposition:
                # Fix V, W. Solve for U.
                # T_reshaped (9 x 9*9) = U * (V * W) ...
                # Let M_r = V[:, r] * W[:, r] (outer product, 9x9)
                # T = sum U[:, r] * M_r
                # This is not a standard matrix product.
                # T is a 3-tensor.
                # We can vectorize T to a vector of length 729.
                # Let t_vec = T.flatten()
                # Let A_r = kron(W[:, r], V[:, r]) ? No.
                # The term is u_r * v_r * w_r.
                # Vectorized: u_r (9) * v_r (9) * w_r (9) -> 729 vector.
                # This is the Kronecker product?
                # If we flatten in C-order (i, j, k):
                # index = i*81 + j*9 + k
                # The contribution of term r is:
                # u_r[i] * v_r[j] * w_r[k]
                # This can be written as:
                # vec(T) = sum_r (w_r * v_r * u_r) (Kronecker product)
                # Specifically, if we define A_r = np.kron(np.kron(w_r, v_r), u_r), then vec(T) = sum A_r.
                # Wait, the order matters.
                # Let's use the property: vec(A * B * C) is not simple.
                # Let's just compute the residual directly for small sizes.
                # 729 is small.
                
                # Update U
                # We want to minimize || T_vec - sum_r A_r(u_r) ||^2
                # Where A_r(u_r) is the vectorized outer product.
                # This is linear in u_r if we fix v_r, w_r.
                # Let M_r = np.kron(w_r, v_r) # 81x81? No.
                # Let's construct the matrix for the linear system.
                # For each r, the contribution to T_vec is:
                # u_r[i] * (v_r[j] * w_r[k])
                # Let P_r[j, k] = v_r[j] * w_r[k] (9x9 matrix)
                # Then T[i, j, k] = sum_r u_r[i] * P_r[j, k]
                # So for each i, T[i, :, :] = sum_r u_r[i] * P_r
                # This is a matrix equation: T_2d[i] = sum_r u_r[i] * P_r
                # Where T_2d[i] is the 9x9 slice of T for index i.
                # We can solve for u_r by stacking the slices.
                # Let T_stacked be a 9x(9*9) matrix? No.
                # For each i (0..8), we have a 9x9 matrix T_i.
                # T_i = sum_r u_r[i] * P_r
                # This is a linear system for u_r[i] for all r.
                # Let U_i be the vector of u_r[i] for all r (length R).
                # Let P_stack be a (9*9) x R matrix, where column r is P_r.flatten().
                # Then T_i.flatten() = P_stack^T * U_i ?
                # T_i.flatten() [j*9+k] = sum_r u_r[i] * P_r[j,k]
                # So T_i.flatten() = P_stack * U_i
                # We solve P_stack * U_i = T_i.flatten()
                
                P_stack = np.zeros((81, target_rank))
                for r in range(target_rank):
                    P_r = np.outer(V[:, r], W[:, r])
                    P_stack[:, r] = P_r.flatten()
                
                # Solve for U
                # U_new[:, r] is the solution for each i?
                # We have 9 systems (one for each i).
                # We can solve them all at once if we stack T_i.
                # T_all = T.reshape(9, 81)
                # U_all = np.linalg.lstsq(P_stack.T, T_all.T, rcond=None)[0].T
                # U_all is 9 x R.
                U_new = np.linalg.lstsq(P_stack.T, T_3x3.reshape(9, 81).T, rcond=None)[0].T
                
                # Update V
                # T[i, j, k] = sum_r v_r[j] * (u_r[i] * w_r[k])
                # For each j, T[:, j, :] = sum_r v_r[j] * Q_r
                # Q_r[i, k] = u_r[i] * w_r[k]
                Q_stack = np.zeros((81, target_rank))
                for r in range(target_rank):
                    Q_r = np.outer(U_new[:, r], W[:, r])
                    Q_stack[:, r] = Q_r.flatten()
                
                V_new = np.linalg.lstsq(Q_stack.T, T_3x3.reshape(9, 9, 9).transpose(1, 0, 2).reshape(9, 81).T, rcond=None)[0].T
                
                # Update W
                # T[i, j, k] = sum_r w_r[k] * (u_r[i] * v_r[j])
                # For each k, T[:, :, k] = sum_r w_r[k] * R_r
                # R_r[i, j] = u_r[i] * v_r[j]
                R_stack = np.zeros((81, target_rank))
                for r in range(target_rank):
                    R_r = np.outer(U_new[:, r], V_new[:, r])
                    R_stack[:, r] = R_r.flatten()
                
                W_new = np.linalg.lstsq(R_stack.T, T_3x3.reshape(9, 9, 9).transpose(0, 1, 2).reshape(81, 9).T, rcond=None)[0].T
                
                # Check convergence
                # Reconstruct T
                T_rec = np.zeros((9, 9, 9))
                for r in range(target_rank):
                    T_rec += np.einsum('i,j,k->ijk', U_new[:, r], V_new[:, r], W_new[:, r])
                error = np.linalg.norm(T_rec - T_3x3)
                
                if error < current_error:
                    current_error = error
                    U, V, W = U_new.copy(), V_new.copy(), W_new.copy()
                
                if current_error < tol:
                    break
                
                # Time check
                if time.time() > deadline - 20:
                    return None

            # Check if solution is close to integers
            # Round to nearest 0.5
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            T_rec_round = np.zeros((9, 9, 9))
            for r in range(target_rank):
                T_rec_round += np.einsum('i,j,k->ijk', U_round[:, r], V_round[:, r], W_round[:, r])
            
            err_round = np.linalg.norm(T_rec_round - T_3x3)
            if err_round < 1e-6:
                if current_error < best_error:
                    best_error = current_error
                    best_solution = (U_round, V_round, W_round)
                    
        return best_solution

    # Try to find a decomposition for 3x3
    inner_decomp = None
    # Try ranks from 23 upwards
    for rank in [23, 24, 25, 26, 27]:
        if time.time() > deadline - 30:
            break
        inner_decomp = find_decomposition(rank, max_iters=200, num_restarts=3)
        if inner_decomp is not None:
            print(f"Found 3x3 decomposition with rank {rank}")
            break
        if time.time() > deadline - 10:
            break

    if inner_decomp is None:
        # Fallback to schoolbook 3x3 (rank 27)
        print("Falling back to schoolbook 3x3")
        U_3x3 = np.zeros((9, 27))
        V_3x3 = np.zeros((9, 27))
        W_3x3 = np.zeros((9, 27))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    a_idx = i * 3 + j
                    b_idx = j * 3 + k
                    c_idx = i * 3 + k
                    U_3x3[a_idx, r] = 1.0
                    V_3x3[b_idx, r] = 1.0
                    W_3x3[c_idx, r] = 1.0
        inner_decomp = (U_3x3, V_3x3, W_3x3)
        inner_rank = 27
    else:
        inner_rank = inner_decomp[0].shape[1]

    # 3. Combine with Strassen outer structure
    # Strassen 2x2 block multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Block indices:
    # A00: 0:3, 0:3 -> A indices 0..8
    # A01: 0:3, 3:6 -> A indices 3..11
    # A10: 3:6, 0:3 -> A indices 12..20
    # A11: 3:6, 3:6 -> A indices 21..29
    
    # B00: 0:3, 0:3 -> B indices 0..8
    # B01: 0:3, 3:6 -> B indices 3..11
    # B10: 3:6, 0:3 -> B indices 12..20
    # B11: 3:6, 3:6 -> B indices 21..29
    
    # C00: 0:3, 0:3 -> C indices 0..8
    # C01: 0:3, 3:6 -> C indices 3..11
    # C10: 3:6, 0:3 -> C indices 12..20
    # C11: 3:6, 3:6 -> C indices 21..29
    
    # W index convention: W[k*n + i, r]
    # For 6x6, n=6.
    # C entry (i, k) corresponds to W index k*6 + i.
    
    # Strassen terms definition:
    # Each term is a list of (block_idx, coeff) for A, B, and C reconstruction.
    # We need to map the inner 3x3 decomposition to the global 6x6 vectors.
    
    # Inner U, V, W are 9 x R_inner.
    # Global U, V, W will be 36 x (7 * R_inner).
    
    total_rank = 7 * inner_rank
    U_global = np.zeros((36, total_rank))
    V_global = np.zeros((36, total_rank))
    W_global = np.zeros((36, total_rank))
    
    # Define Strassen multipliers
    # M1
    a_coeffs_1 = [(0, 1), (3, 1)] # A00, A11
    b_coeffs_1 = [(0, 1), (3, 1)] # B00, B11
    c_coeffs_1 = [(0, 1), (3, -1), (2, 1), (1, -1)] # C00, C11, C10, C01 ? 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the contribution of each M_s to each C_block.
    # C00: M1(+1), M4(+1), M5(-1), M7(+1)
    # C01: M3(+1), M5(+1)
    # C10: M2(+1), M4(+1)
    # C11: M1(+1), M2(-1), M3(+1), M6(+1)
    
    strassen_terms = [
        # (a_blocks, b_blocks, c_contributions)
        # c_contributions: dict {c_block_idx: coeff}
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], {0: 1, 3: 1, 2: -1, 1: 1}), # M1
        ([(2, 1), (3, 1)], [(0, 1)], {2: 1, 3: -1}), # M2
        ([(0, 1)], [(1, 1), (3, -1)], {1: 1, 3: 1}), # M3
        ([(3, 1)], [(2, 1), (0, -1)], {0: 1, 2: 1}), # M4
        ([(0, 1), (1, 1)], [(3, 1)], {0: -1, 1: 1}), # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], {3: 1}), # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], {0: 1}), # M7
    ]
    
    U_3, V_3, W_3 = inner_decomp
    
    for s_idx, (a_blocks, b_blocks, c_contribs) in enumerate(strassen_terms):
        for t in range(inner_rank):
            r_global = s_idx * inner_rank + t
            
            # Fill U_global
            for a_block, a_coeff in a_blocks:
                # a_block is 0,1,2,3 corresponding to A00, A01, A10, A11
                # Map block to global A indices
                # A00: 0..8, A01: 3..11, A10: 12..20, A11: 21..29
                # Wait, A01 is row 0, col 3..5. Indices: 0*6+3=3 to 0*6+5=5?
                # No, A is 6x6.
                # A00: rows 0-2, cols 0-2. Indices: 0*6+0=0 to 2*6+2=14?
                # No, index = row*6 + col.
                # A00: rows 0,1,2; cols 0,1,2.
                # Indices: 0,1,2, 6,7,8, 12,13,14.
                # My previous mapping was wrong.
                # Let's re-calculate block indices.
                
                # Block (br, bc) of size 3x3.
                # Rows: br*3 to br*3+2
                # Cols: bc*3 to bc*3+2
                # Global index: row*6 + col
                
                # A00: br=0, bc=0. Rows 0-2, Cols 0-2.
                # Indices: 0,1,2, 6,7,8, 12,13,14.
                # A01: br=0, bc=1. Rows 0-2, Cols 3-5.
                # Indices: 3,4,5, 9,10,11, 15,16,17.
                # A10: br=1, bc=0. Rows 3-5, Cols 0-2.
                # Indices: 18,19,20, 24,25,26, 30,31,32.
                # A11: br=1, bc=1. Rows 3-5, Cols 3-5.
                # Indices: 21,22,23, 27,28,29, 33,34,35.
                
                # Helper to get indices for a block
                def get_block_indices(br, bc):
                    indices = []
                    for i in range(3):
                        for j in range(3):
                            row = br * 3 + i
                            col = bc * 3 + j
                            indices.append(row * 6 + col)
                    return indices
                
                a_indices = get_block_indices(a_block // 2, a_block % 2)
                # a_block 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                
                for idx_local, idx_global in zip(range(9), a_indices):
                    U_global[idx_global, r_global] += a_coeff * U_3[idx_local, t]
                    
            # Fill V_global
            for b_block, b_coeff in b_blocks:
                b_indices = get_block_indices(b_block // 2, b_block % 2)
                for idx_local, idx_global in zip(range(9), b_indices):
                    V_global[idx_global, r_global] += b_coeff * V_3[idx_local, t]
                    
            # Fill W_global
            # W index: k*6 + i (col*6 + row)
            # C block (cr, cc).
            # C entry (i, k) in global.
            # i is row, k is col.
            # W index is k*6 + i.
            # Inner W_3 index is c_idx = i_local*3 + k_local.
            # i_local is row in 3x3, k_local is col in 3x3.
            
            for c_block, c_coeff in c_contribs.items():
                cr = c_block // 2
                cc = c_block % 2
                # For each entry in this 3x3 block
                for i_local in range(3):
                    for k_local in range(3):
                        c_idx_local = i_local * 3 + k_local
                        val = W_3[c_idx_local, t]
                        if val != 0:
                            row_global = cr * 3 + i_local
                            col_global = cc * 3 + k_local
                            w_index = col_global * 6 + row_global
                            W_global[w_index, r_global] += c_coeff * val

    # Verify
    T_global = np.zeros((36, 36, 36), dtype=np.float64)
    for i in range(36):
        row_a = i // 6
        col_a = i % 6
        for j in range(36):
            row_b = j // 6
            col_b = j % 6
            if col_a == row_b:
                k = row_a * 6 + col_b
                T_global[i, j, k] = 1.0
                
    T_rec = np.zeros((36, 36, 36), dtype=np.float64)
    for r in range(total_rank):
        T_rec += np.einsum('i,j,k->ijk', U_global[:, r], V_global[:, r], W_global[:, r])
        
    if not np.allclose(T_rec, T_global, atol=1e-6):
        print("Verification failed, falling back to 189")
        # Re-run with schoolbook 3x3
        U_3, V_3, W_3 = np.zeros((9, 27)), np.zeros((9, 27)), np.zeros((9, 27))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U_3[i * 3 + j, r] = 1.0
                    V_3[j * 3 + k, r] = 1.0
                    W_3[i * 3 + k, r] = 1.0
        inner_rank = 27
        total_rank = 7 * 27
        U_global = np.zeros((36, total_rank))
        V_global = np.zeros((36, total_rank))
        W_global = np.zeros((36, total_rank))
        
        for s_idx, (a_blocks, b_blocks, c_contribs) in enumerate(strassen_terms):
            for t in range(inner_rank):
                r_global = s_idx * inner_rank + t
                for a_block, a_coeff in a_blocks:
                    a_indices = get_block_indices(a_block // 2, a_block % 2)
                    for idx_local, idx_global in zip(range(9), a_indices):
                        U_global[idx_global, r_global] += a_coeff * U_3[idx_local, t]
                for b_block, b_coeff in b_blocks:
                    b_indices = get_block_indices(b_block // 2, b_block % 2)
                    for idx_local, idx_global in zip(range(9), b_indices):
                        V_global[idx_global, r_global] += b_coeff * V_3[idx_local, t]
                for c_block, c_coeff in c_contribs.items():
                    cr = c_block // 2
                    cc = c_block % 2
                    for i_local in range(3):
                        for k_local in range(3):
                            c_idx_local = i_local * 3 + k_local
                            val = W_3[c_idx_local, t]
                            if val != 0:
                                row_global = cr * 3 + i_local
                                col_global = cc * 3 + k_local
                                w_index = col_global * 6 + row_global
                                W_global[w_index, r_global] += c_coeff * val

    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
