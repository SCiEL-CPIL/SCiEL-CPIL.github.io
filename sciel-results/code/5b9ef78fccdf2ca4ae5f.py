import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 130  # Leave 20s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def build_matrices(U_list, V_list, W_list):
        rank = len(U_list)
        U_mat = np.zeros((n*m, rank))
        V_mat = np.zeros((m*p, rank))
        W_mat = np.zeros((p*n, rank))
        
        for r in range(rank):
            for idx, val in enumerate(U_list[r]):
                if val != 0:
                    U_mat[idx, r] = val
            for idx, val in enumerate(V_list[r]):
                if val != 0:
                    V_mat[idx, r] = val
            for idx, val in enumerate(W_list[r]):
                if val != 0:
                    W_mat[idx, r] = val
        return U_mat, V_mat, W_mat

    def verify(U_mat, V_mat, W_mat):
        U_3d = U_mat.reshape(n, m, -1)
        V_3d = V_mat.reshape(m, p, -1)
        W_3d = W_mat.reshape(p, n, -1)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
                    if not np.isclose(val, 1.0, atol=1e-10):
                        return False
        return True

    # --- 1+2+2 Decomposition (Rank 117) ---
    def get_1_2_2_decomp():
        U = []
        V = []
        W = []
        
        def add_term(u, v, w):
            U.append(u)
            V.append(v)
            W.append(w)
            
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        def add_standard_block(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end + 1):
                for j in range(j_start, j_end + 1):
                    for k in range(k_start, k_end + 1):
                        u = [0] * (n*m)
                        v = [0] * (m*p)
                        w = [0] * (p*n)
                        u[u_idx(i, j)] = 1
                        v[v_idx(j, k)] = 1
                        w[w_idx(k, i)] = 1
                        add_term(u, v, w)

        def add_strassen_2x2x2(i0, j0, k0):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
            v[v_idx(j0, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
            w[w_idx(k0, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
            add_term(u, v, w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0+1, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
            v[v_idx(j0, k0)] = 1
            w[w_idx(k0, i0+1)] = 1; w[w_idx(k0+1, i0+1)] = -1
            add_term(u, v, w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0)] = 1
            v[v_idx(j0, k0+1)] = 1; v[v_idx(j0+1, k0+1)] = -1
            w[w_idx(k0+1, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
            add_term(u, v, w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0+1, j0+1)] = 1
            v[v_idx(j0+1, k0)] = 1; v[v_idx(j0, k0)] = -1
            w[w_idx(k0, i0)] = 1; w[w_idx(k0, i0+1)] = 1
            add_term(u, v, w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0)] = 1; u[u_idx(i0, j0+1)] = 1
            v[v_idx(j0+1, k0+1)] = 1
            w[w_idx(k0, i0)] = -1; w[w_idx(k0+1, i0)] = 1
            add_term(u, v, w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0+1, j0)] = 1; u[u_idx(i0, j0)] = -1
            v[v_idx(j0, k0)] = 1; v[v_idx(j0, k0+1)] = 1
            w[w_idx(k0+1, i0+1)] = 1
            add_term(u, v, w)
            
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0+1)] = 1; u[u_idx(i0+1, j0+1)] = -1
            v[v_idx(j0+1, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
            w[w_idx(k0, i0)] = 1
            add_term(u, v, w)

        blocks = [[0], [1, 2], [3, 4]]
        
        for bi, ib in enumerate(blocks):
            for bj, jb in enumerate(blocks):
                for bk, kb in enumerate(blocks):
                    i_start, i_end = min(ib), max(ib)
                    j_start, j_end = min(jb), max(jb)
                    k_start, k_end = min(kb), max(kb)
                    
                    i_size = len(ib)
                    j_size = len(jb)
                    k_size = len(kb)
                    
                    if i_size == 2 and j_size == 2 and k_size == 2:
                        add_strassen_2x2x2(i_start, j_start, k_start)
                    else:
                        add_standard_block(i_start, i_end, j_start, j_end, k_start, k_end)
                        
        return U, V, W

    # --- ALS-based tensor compression for 3x3x3 ---
    def als_3x3x3(rank, max_iters=200, tol=1e-8):
        """Try to find a rank-r decomposition of the 3x3x3 matrix multiplication tensor."""
        # The 3x3x3 tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}
        # We want T = sum_{r=0}^{rank-1} U[:, :, r] * V[:, :, r] * W[:, :, r]
        # where U is 3x3, V is 3x3, W is 3x3
        
        rng = np.random.default_rng(42)
        U = rng.standard_normal((3, 3, rank))
        V = rng.standard_normal((3, 3, rank))
        W = rng.standard_normal((3, 3, rank))
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if elapsed() > DEADLINE:
                break
            
            # Update U: for each r, solve for U[:,:,r]
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # For fixed r, we want U[:,:,r] to minimize sum_{i,j,k} (T[i,j,k] - sum_{r'} U[i,j,r']*V[j,k,r']*W[k,i,r'])^2
            # This is a linear system in U[:,:,r] for each (i,j)
            
            for r in range(rank):
                # Compute residual with current U[:,:,r] zeroed out
                residual = np.ones((3, 3, 3))  # T is all ones
                for r2 in range(rank):
                    if r2 == r:
                        continue
                    # Contract V[:,:,r2] and W[:,:,r2]
                    # For each (i,j,k): V[j,k,r2] * W[k,i,r2]
                    temp = np.zeros((3, 3, 3))
                    for j in range(3):
                        for k in range(3):
                            for i in range(3):
                                temp[i, j, k] = V[j, k, r2] * W[k, i, r2]
                    residual -= temp
                
                # Now solve for U[:,:,r]: minimize sum_{i,j,k} (residual[i,j,k] - U[i,j,r]*V[j,k,r]*W[k,i,r])^2
                # For each (i,j), this is a linear least squares in U[i,j,r]
                # Actually, we can vectorize: for each (i,j), the coefficient of U[i,j,r] is sum_k V[j,k,r]*W[k,i,r]
                # Wait, that's not right. Let me think again.
                
                # The term is U[i,j,r] * V[j,k,r] * W[k,i,r] for each k
                # So for fixed (i,j), we have: sum_k U[i,j,r] * V[j,k,r] * W[k,i,r] = U[i,j,r] * sum_k V[j,k,r] * W[k,i,r]
                # But that's only if we're looking at a single k. Actually no, each (i,j,k) has its own equation.
                
                # Let me set up the linear system properly.
                # For each (i,j,k): residual[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
                # This is underdetermined if we have 27 equations and 9 unknowns (U[0,0,r], U[0,1,r], ..., U[2,2,r])
                # But each equation involves only one U[i,j,r].
                # So for each (i,j), we have 3 equations (one for each k):
                #   residual[i,j,0] = U[i,j,r] * V[j,0,r] * W[0,i,r]
                #   residual[i,j,1] = U[i,j,r] * V[j,1,r] * W[1,i,r]
                #   residual[i,j,2] = U[i,j,r] * V[j,2,r] * W[2,i,r]
                # Least squares: U[i,j,r] = (sum_k residual[i,j,k] * V[j,k,r] * W[k,i,r]) / (sum_k (V[j,k,r] * W[k,i,r])^2)
                
                for i in range(3):
                    for j in range(3):
                        num = 0.0
                        den = 0.0
                        for k in range(3):
                            coeff = V[j, k, r] * W[k, i, r]
                            num += residual[i, j, k] * coeff
                            den += coeff * coeff
                        if den > 1e-15:
                            U[i, j, r] = num / den
                        else:
                            U[i, j, r] = 0.0
            
            # Update V similarly
            for r in range(rank):
                residual = np.ones((3, 3, 3))
                for r2 in range(rank):
                    if r2 == r:
                        continue
                    temp = np.zeros((3, 3, 3))
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                temp[i, j, k] = U[i, j, r2] * W[k, i, r2]
                    residual -= temp
                
                for j in range(3):
                    for k in range(3):
                        num = 0.0
                        den = 0.0
                        for i in range(3):
                            coeff = U[i, j, r] * W[k, i, r]
                            num += residual[i, j, k] * coeff
                            den += coeff * coeff
                        if den > 1e-15:
                            V[j, k, r] = num / den
                        else:
                            V[j, k, r] = 0.0
            
            # Update W similarly
            for r in range(rank):
                residual = np.ones((3, 3, 3))
                for r2 in range(rank):
                    if r2 == r:
                        continue
                    temp = np.zeros((3, 3, 3))
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                temp[i, j, k] = U[i, j, r2] * V[j, k, r2]
                    residual -= temp
                
                for k in range(3):
                    for i in range(3):
                        num = 0.0
                        den = 0.0
                        for j in range(3):
                            coeff = U[i, j, r] * V[j, k, r]
                            num += residual[i, j, k] * coeff
                            den += coeff * coeff
                        if den > 1e-15:
                            W[k, i, r] = num / den
                        else:
                            W[k, i, r] = 0.0
            
            # Compute error
            err = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        val = 0.0
                        for r in range(rank):
                            val += U[i, j, r] * V[j, k, r] * W[k, i, r]
                        err += (1.0 - val) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
        
        return best_err, best_U, best_V, best_W

    # --- 3+1+1 Decomposition using ALS-found 3x3x3 ---
    def get_3_1_1_decomp(als_result):
        err, U33, V33, W33 = als_result
        rank_333 = U33.shape[2]
        
        U = []
        V = []
        W = []
        
        def add_term(u, v, w):
            U.append(u)
            V.append(v)
            W.append(w)
            
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # The 3+1+1 split: indices {0,1,2} for the 3x3x3 block, {3}, {4}
        # Blocks: 3x3x3, 3x3x1, 3x1x3, 1x3x3, 3x1x1, 1x3x1, 1x1x3, 3x1x1 (wait, need to enumerate all)
        
        # Actually, the split is: i in {0,1,2} or {3} or {4}, same for j and k
        # So we have 3x3x3, 3x3x1, 3x1x3, 1x3x3, 3x1x1, 1x3x1, 1x1x3, 1x1x1 blocks
        
        # For the 3x3x3 block (i,j,k in {0,1,2}), use the ALS result
        for r in range(rank_333):
            u = [0] * (n*m)
            v = [0] * (m*p)
            w = [0] * (p*n)
            for i in range(3):
                for j in range(3):
                    if abs(U33[i, j, r]) > 1e-15:
                        u[u_idx(i, j)] = U33[i, j, r]
            for j in range(3):
                for k in range(3):
                    if abs(V33[j, k, r]) > 1e-15:
                        v[v_idx(j, k)] = V33[j, k, r]
            for k in range(3):
                for i in range(3):
                    if abs(W33[k, i, r]) > 1e-15:
                        w[w_idx(k, i)] = W33[k, i, r]
            add_term(u, v, w)
        
        # For all other blocks, use standard 1-term decomposition
        def add_standard_block(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end + 1):
                for j in range(j_start, j_end + 1):
                    for k in range(k_start, k_end + 1):
                        u = [0] * (n*m)
                        v = [0] * (m*p)
                        w = [0] * (p*n)
                        u[u_idx(i, j)] = 1
                        v[v_idx(j, k)] = 1
                        w[w_idx(k, i)] = 1
                        add_term(u, v, w)
        
        # All blocks except the 3x3x3 one
        # i in {0,1,2}, j in {0,1,2}, k in {3}: 3x3x1
        add_standard_block(0, 2, 0, 2, 3, 3)
        # i in {0,1,2}, j in {0,1,2}, k in {4}: 3x3x1
        add_standard_block(0, 2, 0, 2, 4, 4)
        # i in {0,1,2}, j in {3}, k in {0,1,2}: 3x1x3
        add_standard_block(0, 2, 3, 3, 0, 2)
        # i in {0,1,2}, j in {3}, k in {4}: 3x1x1
        add_standard_block(0, 2, 3, 3, 4, 4)
        # i in {0,1,2}, j in {4}, k in {0,1,2}: 3x1x3
        add_standard_block(0, 2, 4, 4, 0, 2)
        # i in {0,1,2}, j in {4}, k in {4}: 3x1x1
        add_standard_block(0, 2, 4, 4, 4, 4)
        # i in {3}, j in {0,1,2}, k in {0,1,2}: 1x3x3
        add_standard_block(3, 3, 0, 2, 0, 2)
        # i in {3}, j in {0,1,2}, k in {4}: 1x3x1
        add_standard_block(3, 3, 0, 2, 4, 4)
        # i in {3}, j in {3}, k in {0,1,2}: 1x1x3
        add_standard_block(3, 3, 3, 3, 0, 2)
        # i in {3}, j in {3}, k in {4}: 1x1x1
        add_standard_block(3, 3, 3, 3, 4, 4)
        # i in {3}, j in {4}, k in {0,1,2}: 1x1x3
        add_standard_block(3, 3, 4, 4, 0, 2)
        # i in {3}, j in {4}, k in {4}: 1x1x1
        add_standard_block(3, 3, 4, 4, 4, 4)
        # i in {4}, j in {0,1,2}, k in {0,1,2}: 1x3x3
        add_standard_block(4, 4, 0, 2, 0, 2)
        # i in {4}, j in {0,1,2}, k in {4}: 1x3x1
        add_standard_block(4, 4, 0, 2, 4, 4)
        # i in {4}, j in {3}, k in {0,1,2}: 1x1x3
        add_standard_block(4, 4, 3, 3, 0, 2)
        # i in {4}, j in {3}, k in {4}: 1x1x1
        add_standard_block(4, 4, 3, 3, 4, 4)
        # i in {4}, j in {4}, k in {0,1,2}: 1x1x3
        add_standard_block(4, 4, 4, 4, 0, 2)
        # i in {4}, j in {4}, k in {4}: 1x1x1
        add_standard_block(4, 4, 4, 4, 4, 4)
        
        return U, V, W

    # Try ALS for 3x3x3 with rank 23
    best_result = None
    for rank_333 in [23, 24, 25]:
        if elapsed() > DEADLINE - 30:
            break
        err, U33, V33, W33 = als_3x3x3(rank_333, max_iters=100)
        if err < 1e-6:
            best_result = (err, U33, V33, W33)
            break
    
    # Build the best decomposition
    if best_result is not None:
        U_list, V_list, W_list = get_3_1_1_decomp(best_result)
    else:
        U_list, V_list, W_list = get_1_2_2_decomp()
    
    U_mat, V_mat, W_mat = build_matrices(U_list, V_list, W_list)
    
    # Final verification
    if not verify(U_mat, V_mat, W_mat):
        # Fallback to 1+2+2 if 3+1+1 fails verification
        U_list, V_list, W_list = get_1_2_2_decomp()
        U_mat, V_mat, W_mat = build_matrices(U_list, V_list, W_list)
        if not verify(U_mat, V_mat, W_mat):
            raise ValueError("All decompositions invalid")
        
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
