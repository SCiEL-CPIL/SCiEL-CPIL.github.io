"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def check_validity_fast(U, V, W, tol=1e-8):
        """Fast validity check using vectorized operations."""
        rank = U.shape[1]
        # For each (i, k): A_ik = U[i*m:(i+1)*m, :], B_ik rows are V[j*p+k, :], w_ik = W[k*n+i, :]
        # Check: (A_ik * w_ik) @ B_ik.T == I_m
        for i in range(n):
            for k in range(p):
                A_ik = U[i*m:(i+1)*m, :]  # (m, rank)
                B_ik = np.array([V[j*p+k, :] for j in range(m)])  # (m, rank)
                w_ik = W[k*n+i, :]  # (rank,)
                result = (A_ik * w_ik[None, :]) @ B_ik.T  # (m, m)
                if np.max(np.abs(result - np.eye(m))) > tol:
                    return False
        return True

    def strassen_2x2():
        """
        Rank-7 tensor decomposition for 2x2 matrix multiplication.
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+g)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[2, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = -1
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def strassen_4x4_4x9():
        """
        Use Strassen-like block decomposition for 4x4 * 4x9.
        Split B into B_left (4x4) and B_right (4x5).
        C_left = A * B_left: Strassen 4x4*4x4, rank 49
        C_right = A * B_right: schoolbook 4x4*4x5, rank 80
        Total: 129
        """
        U2, V2, W2 = strassen_2x2()
        
        U_left = np.zeros((16, 49))
        V_left = np.zeros((16, 49))
        W_left = np.zeros((16, 49))
        
        def block_indices(bi, bj, dim=4):
            return [(bi*2+lr)*dim + (bj*2+lc) for lr in range(2) for lc in range(2)]
        
        A_blocks = {
            0: [(0,0,1), (1,1,1)],
            1: [(1,0,1), (1,1,1)],
            2: [(0,0,1)],
            3: [(1,1,1)],
            4: [(0,0,1), (0,1,1)],
            5: [(1,0,1), (0,0,-1)],
            6: [(0,1,1), (1,1,-1)]
        }
        B_blocks = {
            0: [(0,0,1), (1,1,1)],
            1: [(0,0,1)],
            2: [(0,1,1), (1,1,-1)],
            3: [(1,0,1), (0,0,-1)],
            4: [(1,1,1)],
            5: [(0,0,1), (0,1,1)],
            6: [(1,0,1), (1,1,1)]
        }
        C_blocks = {
            0: [(0,0,1), (1,1,1)],
            1: [(1,0,1), (1,1,-1)],
            2: [(0,1,1), (1,1,1)],
            3: [(0,0,1), (1,0,1)],
            4: [(0,0,-1), (0,1,1)],
            5: [(1,1,-1)],
            6: [(0,0,1)]
        }
        
        for k in range(7):
            for s in range(7):
                col = k * 7 + s
                for (bi, bj, coeff) in A_blocks[k]:
                    idxs = block_indices(bi, bj, dim=4)
                    for l in range(4):
                        if U2[l, s] != 0:
                            U_left[idxs[l], col] += coeff * U2[l, s]
                for (bk, bj, coeff) in B_blocks[k]:
                    idxs = block_indices(bk, bj, dim=4)
                    for l in range(4):
                        if V2[l, s] != 0:
                            V_left[idxs[l], col] += coeff * V2[l, s]
                for (bi, bj, coeff) in C_blocks[k]:
                    idxs = block_indices(bi, bj, dim=4)
                    for l in range(4):
                        if W2[l, s] != 0:
                            W_left[idxs[l], col] += coeff * W2[l, s]
        
        # Schoolbook for right part: 4x4 * 4x5, rank 80
        U_right = np.zeros((16, 80))
        V_right = np.zeros((20, 80))
        W_right = np.zeros((20, 80))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_right[i*4+j, col] = 1
                    V_right[j*5+k, col] = 1
                    W_right[k*4+i, col] = 1
                    col += 1
        
        rank_total = 49 + 80
        U_full = np.zeros((16, rank_total))
        V_full = np.zeros((36, rank_total))
        W_full = np.zeros((36, rank_total))
        
        U_full[:, :49] = U_left
        for j in range(4):
            for k in range(4):
                V_full[j*9+k, :49] = V_left[j*4+k, :]
        for k in range(4):
            for i in range(4):
                W_full[k*4+i, :49] = W_left[k*4+i, :]
        
        U_full[:, 49:] = U_right
        for j in range(4):
            for k in range(5):
                V_full[j*9+4+k, 49:] = V_right[j*5+k, :]
        for k in range(5):
            for i in range(4):
                W_full[k*4+i, 49:] = W_right[k*4+i, :]
        
        assert check_validity_fast(U_full, V_full, W_full), "Combined 4x4*4x9 failed"
        return U_full, V_full, W_full, rank_total

    def try_random_rank(r, num_restarts=20):
        """Try to find a valid decomposition with rank r using random U,V and solving for W."""
        if r < 36:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(restart * 7919 + r * 131)
            # Use sparse random matrices for better conditioning
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            density = max(0.3, 1.0 / np.sqrt(r) * 2)
            for _ in range(int(n * m * r * density)):
                U[np.random.randint(n * m), np.random.randint(r)] = np.random.choice([-1, 1])
            for _ in range(int(m * p * r * density)):
                V[np.random.randint(m * p), np.random.randint(r)] = np.random.choice([-1, 1])
            
            # Precompute: for each w, build the linear system for W[w,:]
            # Equation for w: sum_c (U[u,c] * V[v,c]) * W[w,c] = T[u,v,w]
            # For fixed w, the (u,v) pairs that have T[u,v,w] != 0 are those where
            # w = k*n + i, and u = i*m + j, v = j*p + k for some j
            # So for w = k*n + i: u = i*m + j, v = j*p + k, and T = 1 for all j
            # For other (u,v) pairs with same w: T = 0
            
            W = np.zeros((p * n, r))
            valid = True
            
            for w in range(p * n):
                i_out = w % n
                k_out = w // n
                
                # Build system: for each (u, v), coefficient vector c_{uv} = U[u,:] * V[v,:]
                # We need W[w,:] dot c_{uv} = T[u,v,w]
                
                # Collect non-zero target equations and zero equations
                A_rows = []
                b_vals = []
                
                for u in range(n * m):
                    i_u = u // m
                    j_u = u % m
                    for v in range(m * p):
                        j_v = v // p
                        k_v = v % p
                        # T[u,v,w] = 1 if w = k_v*n + i_u and j_u == j_v, else 0
                        if j_u == j_v and k_v == k_out and i_u == i_out:
                            target = 1.0
                        else:
                            target = 0.0
                        coeff = U[u, :] * V[v, :]
                        # Only add if coeff is non-zero
                        if np.any(coeff != 0):
                            A_rows.append(coeff)
                            b_vals.append(target)
                
                if len(A_rows) == 0:
                    continue
                    
                A_mat = np.array(A_rows)  # (num_eq, r)
                b_vec = np.array(b_vals)  # (num_eq,)
                
                try:
                    W[w, :], residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    valid = False
                    break
            
            if not valid:
                continue
            
            if check_validity_fast(U, V, W):
                return U, V, W, r
        
        return None

    # Get schoolbook solution as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try Strassen-based decomposition (rank 129)
    if time.time() < deadline - 30:
        try:
            U_str, V_str, W_str, rank_str = strassen_4x4_4x9()
            if rank_str < best_rank:
                best_U, best_V, best_W, best_rank = U_str, V_str, W_str, rank_str
        except Exception:
            pass
    
    # Try lower ranks via optimized random search
    # Focus on ranks just below 129 that might be achievable
    search_ranks = [128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
    for r in search_ranks:
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            continue
        
        result = try_random_rank(r, num_restarts=5)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
