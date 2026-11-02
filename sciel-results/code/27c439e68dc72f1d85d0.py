import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 28 Strassen solution
    def get_rank_28_solution():
        rank = 28
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        U_c = [
            {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1}, {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
        ]
        V_c = [
            {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1}, {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
        ]
        W_c = [
            {0: 1, 3: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 1: 1}, {0: -1, 2: 1}, {3: 1}, {0: 1}
        ]
        
        for q in range(4):
            for s in range(7):
                r = q * 7 + s
                for u_idx, val in U_c[s].items():
                    U[u_idx, r] = val
                for v_local, val in V_c[s].items():
                    j = v_local // 2
                    k_loc = v_local % 2
                    k = 2*q + k_loc
                    v_idx = j * p + k
                    V[v_idx, r] = val
                for w_local, val in W_c[s].items():
                    i = w_local % 2
                    k_loc = w_local // 2
                    k = 2*q + k_loc
                    w_idx = k * n + i
                    W[w_idx, r] = val
        return {"U": U, "V": V, "W": W}

    # Helper to verify a solution
    def verify_solution(U, V, W):
        # U: (4, R), V: (16, R), W: (16, R)
        # Tensor T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k',i',r] ? 
        # No, the convention is:
        # U: (n*m, R) -> index (i*m + j) ? No, U is (n*m, R). 
        # The tensor for matmul A (n x m) * B (m x p) -> C (n x p)
        # T[i, j, k] is 1 if C[i,k] depends on A[i,j] and B[j,k].
        # Wait, the standard tensor T has indices (i, j, k) where:
        # A is n x m, B is m x p, C is n x p.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # So T[i, j, k] = 1.
        # 
        # The factorization is: T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[i, k, r]? 
        # No, the interface specifies:
        # U: (n*m, rank) -> U[i*m+j, r]? Or U[i, j, r] flattened?
        # The prompt says: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # And "W[k*n+i,r]" trace convention.
        # 
        # Let's check the indices in the current program.
        # U[u_idx, r] where u_idx is 0..3 (A indices).
        # V[v_idx, r] where v_idx is j*p + k.
        # W[w_idx, r] where w_idx is k*n + i.
        # 
        # The tensor entry T[i, j, k] should be reconstructed as:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ?
        # Wait, W is (p*n, rank). The index is k*n + i.
        # The term in the sum for C[i,k] is A[i,j] B[j,k].
        # So the tensor element is 1 at (i, j, k).
        # The reconstruction must yield 1 for all (i, j, k) and 0 elsewhere?
        # No, the tensor is defined such that T[i, j, k] = 1.
        # The factorization approximates the tensor.
        # 
        # Let's verify by computing the matrix product for random A, B.
        # C = A @ B
        # C_calc[i, k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k]) * W[k*n+i, r] ??
        # This doesn't look right. The rank-1 term is U_vec * V_vec * W_vec.
        # U_vec has non-zeros at A indices. V_vec at B indices. W_vec at C indices.
        # The contribution to C[i,k] from term r is:
        # (linear comb of A) * (linear comb of B) * W[i,k,r]?
        # 
        # Let's look at the Strassen construction in the code.
        # U_c[s] is a dict of A indices.
        # V_c[s] is a dict of B indices.
        # W_c[s] is a dict of C indices.
        # 
        # For a fixed r, let $M_r = (\sum_{u} U[u,r] A[u]) * (\sum_{v} V[v,r] B[v])$.
        # This is a scalar.
        # Then the contribution to C is $M_r * W[:,r]$?
        # No, W is (p*n, rank).
        # 
        # Actually, the standard tensor decomposition for matmul is:
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
        # But here the shapes are flattened.
        # U is (n*m, R). Index i*m+j.
        # V is (m*p, R). Index j*p+k.
        # W is (p*n, R). Index k*n+i.
        # 
        # So T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # 
        # Let's test this verification logic.
        
        R = U.shape[1]
        # Create the target tensor T of shape (n, m, p)
        T_target = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i, j, k] = 1.0
        
        # Reconstruct T
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            # Extract vectors
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            w_vec = W[:, r].reshape(p, n) # W index k*n+i -> reshape to (p, n)
            
            # The term is U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This is an outer product of three tensors?
            # T[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
            
            # Efficient computation:
            # For each j:
            #   term_j = u_vec[:, j] * v_vec[j, :] # shape (n, p)
            #   This term is multiplied by W[k,i,r] which depends on k and i.
            #   So T[i,j,k] += u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
            
            # We can compute this by iterating j
            for j in range(m):
                # u_part: (n,)
                # v_part: (p,)
                # w_part: (p, n)
                # Contribution to T[:, j, :] is:
                # sum_i,k u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
                # This is not a simple outer product because W mixes i and k.
                
                # Let's just do the triple loop for verification speed (n=2, m=2, p=8 is small)
                for i in range(n):
                    for k in range(p):
                        T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        err = np.linalg.norm(T_target - T_recon)
        return err

    # Try to find Rank 24 solution
    def find_rank_24_solution():
        # We decompose into two 2x2x4 blocks
        # Block 1: cols 0-3
        # Block 2: cols 4-7
        
        # We need a rank 12 decomposition for 2x2x4
        # Let's try to search for it
        target_rank = 12
        n_local, m_local, p_local = 2, 2, 4
        
        # Target tensor for 2x2x4
        T_target_local = np.zeros((n_local, m_local, p_local))
        for i in range(n_local):
            for j in range(m_local):
                for k in range(p_local):
                    T_target_local[i, j, k] = 1.0
        
        # Initialize with Strassen (rank 7) + 5 random terms? 
        # Or just random initialization
        np.random.seed(42)
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Use a simple gradient descent or Nelder-Mead
        # Given the small size, we can try a few random restarts
        
        for restart in range(20):
            if time.time() > deadline - 30: break
            
            U_loc = np.random.randn(n_local*m_local, target_rank)
            V_loc = np.random.randn(m_local*p_local, target_rank)
            W_loc = np.random.randn(p_local*n_local, target_rank)
            
            # Scale to help convergence
            U_loc *= 1.0
            V_loc *= 1.0
            W_loc *= 1.0
            
            # Optimize using scipy minimize (Nelder-Mead)
            # Flatten parameters
            params = np.concatenate([U_loc.flatten(), V_loc.flatten(), W_loc.flatten()])
            
            def objective(params):
                U_f = params[:n_local*m_local*target_rank].reshape(n_local, m_local, target_rank)
                V_f = params[n_local*m_local*target_rank : n_local*m_local*target_rank + m_local*p_local*target_rank].reshape(m_local, p_local, target_rank)
                W_f = params[n_local*m_local*target_rank + m_local*p_local*target_rank :].reshape(p_local, n_local, target_rank)
                
                T_recon = np.zeros((n_local, m_local, p_local))
                for i in range(n_local):
                    for j in range(m_local):
                        for k in range(p_local):
                            # Sum over r
                            # U[i,j,r] * V[j,k,r] * W[k,i,r]
                            T_recon[i, j, k] = np.sum(U_f[i, j, :] * V_f[j, k, :] * W_f[k, i, :])
                
                err = np.linalg.norm(T_target_local - T_recon)
                return err
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, params, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception as e:
                pass
        
        if best_err < 1e-6:
            # Extract best solution
            U_flat, V_flat, W_flat = best_params[:n_local*m_local*target_rank], best_params[n_local*m_local*target_rank : n_local*m_local*target_rank + m_local*p_local*target_rank], best_params[n_local*m_local*target_rank + m_local*p_local*target_rank :]
            U_loc_best = U_flat.reshape(n_local, m_local, target_rank)
            V_loc_best = V_flat.reshape(m_local, p_local, target_rank)
            W_loc_best = W_flat.reshape(p_local, n_local, target_rank)
            
            # Construct full 2x2x8 solution
            U_full = np.zeros((n*m, 24))
            V_full = np.zeros((m*p, 24))
            W_full = np.zeros((p*n, 24))
            
            # Block 1 (cols 0-3)
            for i in range(n):
                for j in range(m):
                    U_full[i*m+j, 0:12] = U_loc_best[i, j, :]
            for j in range(m):
                for k in range(p_local):
                    V_full[j*p + k, 0:12] = V_loc_best[j, k, :]
            for k in range(p_local):
                for i in range(n):
                    W_full[k*n + i, 0:12] = W_loc_best[k, i, :]
                    
            # Block 2 (cols 4-7)
            for i in range(n):
                for j in range(m):
                    U_full[i*m+j, 12:24] = U_loc_best[i, j, :]
            for j in range(m):
                for k in range(p_local):
                    V_full[j*p + (k + p_local), 12:24] = V_loc_best[j, k, :]
            for k in range(p_local):
                for i in range(n):
                    W_full[(k + p_local)*n + i, 12:24] = W_loc_best[k, i, :]
            
            # Verify full solution
            err_full = verify_solution(U_full, V_full, W_full)
            if err_full < 1e-6:
                return {"U": U_full, "V": V_full, "W": W_full}
            
        return None

    # Try Rank 24
    if time.time() < deadline - 10:
        sol_24 = find_rank_24_solution()
        if sol_24:
            return sol_24
            
    # Fallback to Rank 28
    return get_rank_28_solution()
