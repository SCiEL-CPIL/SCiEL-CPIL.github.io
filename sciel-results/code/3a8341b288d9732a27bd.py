import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 8
    
    # Build the target tensor
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a rank-10 decomposition for the 2x3x2 block
    T_block = np.zeros((6, 6, 4), dtype=np.int64)
    for i in range(2):
        for j in range(3):
            for k in range(2):
                T_block[i*3+j, j*2+k, k*2+i] = 1
    
    # Try to find low-rank solutions for the block using improved ALS
    np.random.seed(42)
    found_rank10 = False
    found_rank9 = False
    found_rank8 = False
    U_r_best = None
    V_r_best = None
    W_r_best = None
    best_block_rank = 10
    
    for target_rank in [8, 9, 10]:
        if time.time() > deadline - 40:
            break
        if found_rank8 or found_rank9:
            break
        
        for trial in range(200):
            if time.time() > deadline - 30:
                break
            
            rank = target_rank
            # Initialize with small random values
            U = np.random.randn(6, rank) * 0.5
            V = np.random.randn(6, rank) * 0.5
            W = np.random.randn(4, rank) * 0.5
            
            # ALS iterations with proper linear solves
            for it in range(100):
                # Precompute matrices
                # For U update: M_vw[b*4+c, d] = V[b,d]*W[c,d], shape (24, rank)
                M_vw = np.zeros((24, rank))
                for d in range(rank):
                    M_vw[:, d] = np.kron(V[:, d], W[:, d])
                
                # For V update: M_uw[a*4+c, d] = U[a,d]*W[c,d], shape (24, rank)
                M_uw = np.zeros((24, rank))
                for d in range(rank):
                    M_uw[:, d] = np.kron(U[:, d], W[:, d])
                
                # For W update: M_uv[a*6+b, d] = U[a,d]*V[b,d], shape (36, rank)
                M_uv = np.zeros((36, rank))
                for d in range(rank):
                    M_uv[:, d] = np.kron(U[:, d], V[:, d])
                
                # Update U
                for a in range(6):
                    t_vec = T_block[a, :, :].reshape(-1)  # (24,)
                    # Solve: U[a,:] @ M_vw = t_vec  =>  M_vw.T @ U[a,:]^T = t_vec
                    try:
                        U[a, :] = np.linalg.lstsq(M_vw.T, t_vec, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b in range(6):
                    t_vec = T_block[:, b, :].reshape(-1)  # (24,)
                    # T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
                    # For fixed b: sum_d V[b,d] * (U[a,d]*W[c,d]) = T[a,b,c]
                    # So: V[b,:] @ M_uw_b = t_vec where M_uw_b[a*4+c, d] = U[a,d]*W[c,d]
                    # But M_uw is already (24, rank) with M_uw[a*4+c, d] = U[a,d]*W[c,d]
                    try:
                        V[b, :] = np.linalg.lstsq(M_uw.T, t_vec, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for c in range(4):
                    t_vec = T_block[:, :, c].reshape(-1)  # (36,)
                    # T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
                    # For fixed c: W[c,:] @ M_uv = t_vec where M_uv[a*6+b, d] = U[a,d]*V[b,d]
                    try:
                        W[c, :] = np.linalg.lstsq(M_uv.T, t_vec, rcond=None)[0]
                    except:
                        pass
                
                # Check convergence
                T_check = np.einsum('ad,bd,cd->abc', U, V, W)
                if np.allclose(T_check, T_block, atol=1e-6):
                    break
            
            # Try rounding to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            T_check = np.einsum('ad,bd,cd->abc', U_r, V_r, W_r)
            if np.array_equal(T_check, T_block):
                if target_rank == 8:
                    found_rank8 = True
                    best_block_rank = 8
                elif target_rank == 9:
                    found_rank9 = True
                    best_block_rank = 9
                else:
                    found_rank10 = True
                    best_block_rank = 10
                U_r_best = U_r
                V_r_best = V_r
                W_r_best = W_r
                break
    
    if U_r_best is not None:
        block_rank = best_block_rank
        total_rank = 4 * block_rank
        U_full = np.zeros((n*m, total_rank), dtype=float)
        V_full = np.zeros((m*p, total_rank), dtype=float)
        W_full = np.zeros((p*n, total_rank), dtype=float)
        
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            offset = block * block_rank
            
            for a in range(6):
                i, j = a // 3, a % 3
                U_full[i*3+j, offset:(offset+block_rank)] = U_r_best[a, :]
            
            for b in range(6):
                j, k = b // 2, b % 2
                k_full = k0 + k
                V_full[j*8+k_full, offset:(offset+block_rank)] = V_r_best[b, :]
            
            for c in range(4):
                k, i = c // 2, c % 2
                k_full = k0 + k
                W_full[k_full*2+i, offset:(offset+block_rank)] = W_r_best[c, :]
        
        T_check_full = np.einsum('ad,bd,cd->abc', U_full, V_full, W_full)
        if np.array_equal(T_check_full, T_exp):
            # Convert to list (values should be half-integers)
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: rank-44 solution
    U = np.zeros((n*m, 44), dtype=float)
    V = np.zeros((m*p, 44), dtype=float)
    W = np.zeros((p*n, 44), dtype=float)
    
    col_offset = 0
    
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Block 1: A[:, 0] * B[0, k0:k1]  (2x1 * 1x2) - rank 4
        U[0*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1
        
        U[0*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        
        U[1*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        
        U[1*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        # Block 2: A[:, 1:3] * B[1:3, k0:k1]  (2x2 * 2x2) - Strassen rank 7
        U[0*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        U[1*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        W[k1*n + 1, col_offset] = -1
        col_offset += 1
        
        U[0*m + 1, col_offset] = 1
        V[1*p + k1, col_offset] = 1
        V[2*p + k1, col_offset] = -1
        W[k1*n + 0, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        U[1*m + 2, col_offset] = 1
        V[2*p + k0, col_offset] = 1
        V[1*p + k0, col_offset] = -1
        W[k0*n + 0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        
        U[0*m + 1, col_offset] = 1
        U[0*m + 2, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = -1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        
        U[1*m + 1, col_offset] = 1
        U[0*m + 1, col_offset] = -1
        V[1*p + k0, col_offset] = 1
        V[1*p + k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        U[0*m + 2, col_offset] = 1
        U[1*m + 2, col_offset] = -1
        V[2*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1

    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.array_equal(T_check, T_exp)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=float)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=float)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, col] = 1
                    V_sb[b_idx, col] = 1
                    W_sb[c_idx, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
