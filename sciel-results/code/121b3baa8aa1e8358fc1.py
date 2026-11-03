# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 75
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for 120s limit
    
    def check_deadline():
        return time.time() > deadline
    
    # Build schoolbook decomposition as fallback
    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-6:
                        return False
        return True

    def find_low_rank_decomp(n_sub, m_sub, p_sub, target_rank, max_iters=100, n_restarts=3):
        """Try to find a rank-target_rank decomposition for n_sub x m_sub x p_sub."""
        if target_rank >= n_sub * m_sub * p_sub:
            # Use schoolbook
            U = [[0] * (n_sub * m_sub * p_sub) for _ in range(n_sub * m_sub)]
            V = [[0] * (n_sub * m_sub * p_sub) for _ in range(m_sub * p_sub)]
            W = [[0] * (n_sub * m_sub * p_sub) for _ in range(p_sub * n_sub)]
            col = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U[i * m_sub + j][col] = 1
                        V[j * p_sub + k][col] = 1
                        W[k * n_sub + i][col] = 1
                        col += 1
            return U, V, W, n_sub * m_sub * p_sub
        
        best_rank = n_sub * m_sub * p_sub
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(n_restarts):
            if check_deadline():
                break
            
            np.random.seed(42 + attempt * 17)
            U = np.random.randn(n_sub * m_sub, target_rank) * 0.5
            V = np.random.randn(m_sub * p_sub, target_rank) * 0.5
            W = np.random.randn(p_sub * n_sub, target_rank) * 0.5
            
            converged = False
            for iter in range(max_iters):
                if check_deadline():
                    break
                
                # Update U
                # A[i*m+j, c] = sum_k V[j*p+k, c] * W[k*n+i, c]
                # Vectorize using einsum
                # V shape: (m_sub * p_sub, target_rank)
                # W shape: (p_sub * n_sub, target_rank)
                # We need to reshape V and W to use einsum
                V_reshaped = V.reshape(m_sub, p_sub, target_rank)
                W_reshaped = W.reshape(p_sub, n_sub, target_rank)
                
                # A[i, j, c] = sum_k V[j, k, c] * W[k, i, c]
                A = np.einsum('jkc,kic->ijc', V_reshaped, W_reshaped)
                A_flat = A.reshape(n_sub * m_sub, target_rank)
                
                for r in range(n_sub * m_sub):
                    if np.linalg.norm(A_flat[r, :]) > 1e-10:
                        U[r, :] = np.linalg.lstsq(A_flat[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                
                # Update V
                # B[j*p+k, c] = sum_i U[i*m+j, c] * W[k*n+i, c]
                U_reshaped = U.reshape(n_sub, m_sub, target_rank)
                W_reshaped = W.reshape(p_sub, n_sub, target_rank)
                
                # B[j, k, c] = sum_i U[i, j, c] * W[k, i, c]
                B = np.einsum('ijc,kic->jk c', U_reshaped, W_reshaped)
                B_flat = B.reshape(m_sub * p_sub, target_rank)
                
                for r in range(m_sub * p_sub):
                    if np.linalg.norm(B_flat[r, :]) > 1e-10:
                        V[r, :] = np.linalg.lstsq(B_flat[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                        
                # Update W
                # C[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                U_reshaped = U.reshape(n_sub, m_sub, target_rank)
                V_reshaped = V.reshape(m_sub, p_sub, target_rank)
                
                # C[k, i, c] = sum_j U[i, j, c] * V[j, k, c]
                C = np.einsum('ijc,jkc->kic', U_reshaped, V_reshaped)
                C_flat = C.reshape(p_sub * n_sub, target_rank)
                
                for r in range(p_sub * n_sub):
                    if np.linalg.norm(C_flat[r, :]) > 1e-10:
                        W[r, :] = np.linalg.lstsq(C_flat[r, :].reshape(1, -1), np.ones(1), rcond=None)[0][0]
                        
                # Check convergence
                err = 0
                U_reshaped = U.reshape(n_sub, m_sub, target_rank)
                V_reshaped = V.reshape(m_sub, p_sub, target_rank)
                W_reshaped = W.reshape(p_sub, n_sub, target_rank)
                
                # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
                # This is a trace-like contraction
                # We can compute it by iterating over i, j, k
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            val = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                            err += (val - 1) ** 2
                
                if err < 1e-6:
                    converged = True
                    break
            
            if converged:
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_U is not None:
            return best_U, best_V, best_W, best_rank
        else:
            # Fall back to schoolbook
            rank = n_sub * m_sub * p_sub
            U = [[0] * rank for _ in range(n_sub * m_sub)]
            V = [[0] * rank for _ in range(m_sub * p_sub)]
            W = [[0] * rank for _ in range(p_sub * n_sub)]
            col = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U[i * m_sub + j][col] = 1
                        V[j * p_sub + k][col] = 1
                        W[k * n_sub + i][col] = 1
                        col += 1
            return U, V, W, rank

    best_total_rank = rank_school
    best_U, best_V, best_W = build_schoolbook()
    
    # Try to find lower rank for 3x3x3
    # Increase restarts and iterations for this critical block
    U4, V4, W4, r4 = find_low_rank_decomp(3, 3, 3, 23, max_iters=100, n_restarts=20)
    
    if r4 < 27:
        total_rank = 12 + 18 + 18 + r4
        if total_rank < best_total_rank:
            U_full = [[0] * total_rank for _ in range(n * m)]
            V_full = [[0] * total_rank for _ in range(m * p)]
            W_full = [[0] * total_rank for _ in range(p * n)]
            
            col = 0
            # Sub-problem 1: 3x2x2
            U1, V1, W1, r1 = find_low_rank_decomp(3, 2, 2, 12, max_iters=10, n_restarts=1)
            for i in range(3):
                for j in range(2):
                    for c in range(r1):
                        U_full[i * m + j][col + c] = U1[i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    for c in range(r1):
                        V_full[j * p + k][col + c] = V1[j * 2 + k][c]
            for k in range(2):
                for i in range(3):
                    for c in range(r1):
                        W_full[k * n + i][col + c] = W1[k * 3 + i][c]
            col += r1
            
            # Sub-problem 2: 3x3x2
            U2, V2, W2, r2 = find_low_rank_decomp(3, 3, 2, 18, max_iters=10, n_restarts=1)
            for i in range(3):
                for j in range(3):
                    for c in range(r2):
                        U_full[i * m + (j + 2)][col + c] = U2[i * 3 + j][c]
            for j in range(3):
                for k in range(2):
                    for c in range(r2):
                        V_full[(j + 2) * p + k][col + c] = V2[j * 2 + k][c]
            for k in range(2):
                for i in range(3):
                    for c in range(r2):
                        W_full[k * n + i][col + c] = W2[k * 3 + i][c]
            col += r2
            
            # Sub-problem 3: 3x2x3
            U3, V3, W3, r3 = find_low_rank_decomp(3, 2, 3, 18, max_iters=10, n_restarts=1)
            for i in range(3):
                for j in range(2):
                    for c in range(r3):
                        U_full[i * m + j][col + c] = U3[i * 2 + j][c]
            for j in range(2):
                for k in range(3):
                    for c in range(r3):
                        V_full[j * p + (k + 2)][col + c] = V3[j * 3 + k][c]
            for k in range(3):
                for i in range(3):
                    for c in range(r3):
                        W_full[(k + 2) * n + i][col + c] = W3[k * 3 + i][c]
            col += r3
            
            # Sub-problem 4: 3x3x3
            for i in range(3):
                for j in range(3):
                    for c in range(r4):
                        U_full[i * m + (j + 2)][col + c] = U4[i * 3 + j][c]
            for j in range(3):
                for k in range(3):
                    for c in range(r4):
                        V_full[(j + 2) * p + (k + 2)][col + c] = V4[j * 3 + k][c]
            for k in range(3):
                for i in range(3):
                    for c in range(r4):
                        W_full[(k + 2) * n + i][col + c] = W4[k * 3 + i][c]
            col += r4
            
            if verify_decomp(U_full, V_full, W_full, total_rank):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_total_rank = total_rank

    if best_total_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        U, V, W = build_schoolbook()
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
