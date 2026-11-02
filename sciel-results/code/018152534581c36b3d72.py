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
    
    # Try rank 10 with random initialization and ALS
    best_rank = 44
    best_U = None
    best_V = None
    best_W = None
    
    # First, let's try to find a rank-10 solution for the block
    # using a random search with integer/half-integer entries
    
    np.random.seed(42)
    found_rank10 = False
    
    for trial in range(100):
        if time.time() > deadline - 30:
            break
        
        # Try rank 10
        rank = 10
        U = np.random.randint(-2, 3, (6, rank)) * 0.5
        V = np.random.randint(-2, 3, (6, rank)) * 0.5
        W = np.random.randint(-2, 3, (4, rank)) * 0.5
        
        # ALS iterations
        for it in range(50):
            # Update U: for each a, U[a,:] = (V^T ⊙ W^T)^{-1} * T[:,a,:] flattened
            for a in range(6):
                # T[:,a,:] is (6,4) matrix, reshape to (24,)
                t_vec = T_block[:, a, :].reshape(-1)
                # (V^T ⊙ W^T) is (24, rank)
                vw = np.kron(V.T, W.T).T  # (rank, 24) -> (24, rank)
                # Actually, we need: sum_d U[a,d] * V[:,d] * W[:,d] = T[:,a,:]
                # This is: U[a,:] @ (V ⊙ W) = T[:,a,:].reshape(-1)
                # where (V ⊙ W) is (24, rank) with (V ⊙ W)[b*n+c, d] = V[b,d]*W[c,d]
                vw_mat = np.zeros((24, rank))
                for d in range(rank):
                    vw_mat[:, d] = np.kron(V[:, d], W[:, d])
                # Solve: U[a,:] @ vw_mat = t_vec
                try:
                    U[a, :] = np.linalg.lstsq(vw_mat.T, t_vec, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b in range(6):
                t_vec = T_block[:, b, :].reshape(-1)
                uw_mat = np.zeros((24, rank))
                for d in range(rank):
                    uw_mat[:, d] = np.kron(U[:, d], W[:, d])
                try:
                    V[b, :] = np.linalg.lstsq(uw_mat.T, t_vec, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c in range(4):
                t_vec = T_block[:, :, c].reshape(-1)
                uv_mat = np.zeros((36, rank))
                for d in range(rank):
                    uv_mat[:, d] = np.kron(U[:, d], V[:, d])
                # T[:, :, c] is (6,6), but we only need entries where T[a,b,c] != 0
                # Actually, the full equation is: sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
                # For fixed c: W[c,:] @ (U ⊙ V) = T[:,:,c].reshape(-1)
                # (U ⊙ V) is (36, rank) with (U ⊙ V)[a*6+b, d] = U[a,d]*V[b,d]
                uv_mat = np.zeros((36, rank))
                for d in range(rank):
                    uv_mat[:, d] = np.kron(U[:, d], V[:, d])
                t_vec = T_block[:, :, c].reshape(-1)
                try:
                    W[c, :] = np.linalg.lstsq(uv_mat.T, t_vec, rcond=None)[0]
                except:
                    pass
        
        # Round to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify
        T_check = np.einsum('ad,bd,cd->abc', U_r, V_r, W_r)
        if np.array_equal(T_check, T_block):
            found_rank10 = True
            break
    
    if found_rank10:
        # Build the full rank-40 solution
        U_full = np.zeros((n*m, 40), dtype=np.int64)
        V_full = np.zeros((m*p, 40), dtype=np.int64)
        W_full = np.zeros((p*n, 40), dtype=np.int64)
        
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            offset = block * 10
            
            # Map block indices to full indices
            for a in range(6):
                i, j = a // 3, a % 3
                U_full[i*3+j, offset:(offset+10)] = U_r[a, :]
            
            for b in range(6):
                j, k = b // 2, b % 2
                k_full = k0 + k
                V_full[j*8+k_full, offset:(offset+10)] = V_r[b, :]
            
            for c in range(4):
                k, i = c // 2, c % 2
                k_full = k0 + k
                W_full[k_full*2+i, offset:(offset+10)] = W_r[c, :]
        
        def verify_full(U, V, W):
            r = U.shape[1]
            T_check = np.einsum('ad,bd,cd->abc', U, V, W)
            return np.array_equal(T_check, T_exp)
        
        if verify_full(U_full, V_full, W_full):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: rank-44 solution
    U = np.zeros((n*m, 44), dtype=np.int64)
    V = np.zeros((m*p, 44), dtype=np.int64)
    W = np.zeros((p*n, 44), dtype=np.int64)
    
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
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.int64)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.int64)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.int64)
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
