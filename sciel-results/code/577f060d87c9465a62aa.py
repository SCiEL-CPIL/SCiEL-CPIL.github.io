import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor entries for 4x5x6
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    target_entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries[(a, b, c)] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank

    def verify_fast(U, V, W, tol=1e-6):
        """Fast verification using einsum."""
        R = np.einsum('ad,bd,cd->abc', U, V, W)
        T = np.zeros((nm, mp, pn))
        for (a, b, c), val in target_entries.items():
            T[a, b, c] = val
        diff = R - T
        if np.max(np.abs(diff)) > tol:
            return False
        return True

    # Fallback to schoolbook
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # Strategy: Decompose into 4 blocks of 2x5x3
    n_sub, m_sub, p_sub = 2, 5, 3
    nm_sub = n_sub * m_sub # 10
    mp_sub = m_sub * p_sub # 15
    pn_sub = p_sub * n_sub # 6
    
    def solve_sub_problem(max_time=30.0, target_rank=15):
        """Solve for 2x5x3 multiplication rank."""
        start = time.time()
        
        # Try target rank
        r = target_rank
        
        # Try a few random initializations
        for seed in range(20):
            if time.time() - start > max_time:
                break
            
            np.random.seed(seed)
            # Scale initialization to help convergence
            U = np.random.randn(nm_sub, r) * 0.5
            V = np.random.randn(mp_sub, r) * 0.5
            W = np.random.randn(pn_sub, r) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() - start > max_time:
                    break
                
                # Update U
                # M_U shape: (mp_sub * pn_sub, r)
                # M_U[(j, k), l] = V[j, l] * W[k, l]
                M_U = np.zeros((mp_sub * pn_sub, r))
                for j in range(m_sub):
                    for k in range(p_sub):
                        idx = j * p_sub + k
                        M_U[idx, :] = V[j, :] * W[k, :]
                
                t_U = np.ones(mp_sub * pn_sub)
                # Solve M_U @ U[a, :] = t_U for each a
                # Since M_U is same for all a, we can solve once and broadcast?
                # No, U[a, :] is the unknown.
                # M_U X = t_U.
                # X = lstsq(M_U, t_U)
                try:
                    sol_U = np.linalg.lstsq(M_U, t_U, rcond=None)[0]
                    # Check if solution is good
                    if np.allclose(M_U @ sol_U, t_U, atol=1e-8):
                        U[:, :] = sol_U
                except:
                    pass

                # Update V
                M_V = np.zeros((nm_sub * pn_sub, r))
                for i in range(n_sub):
                    for k in range(p_sub):
                        idx = i * p_sub + k
                        M_V[idx, :] = U[i, :] * W[k, :]
                
                t_V = np.ones(nm_sub * pn_sub)
                try:
                    sol_V = np.linalg.lstsq(M_V, t_V, rcond=None)[0]
                    if np.allclose(M_V @ sol_V, t_V, atol=1e-8):
                        V[:, :] = sol_V
                except:
                    pass

                # Update W
                M_W = np.zeros((nm_sub * mp_sub, r))
                for i in range(n_sub):
                    for j in range(m_sub):
                        idx = i * m_sub + j
                        M_W[idx, :] = U[i, :] * V[j, :]
                
                t_W = np.ones(nm_sub * mp_sub)
                try:
                    sol_W = np.linalg.lstsq(M_W, t_W, rcond=None)[0]
                    if np.allclose(M_W @ sol_W, t_W, atol=1e-8):
                        W[:, :] = sol_W
                except:
                    pass
                
                # Check convergence
                # Compute residual
                R_sub = np.einsum('ad,bd,cd->abc', U, V, W)
                T_sub = np.ones((nm_sub, mp_sub, pn_sub))
                # Note: T_sub indices are (i, j, k) in sub-space?
                # Our U, V, W are indexed by (a, d), (b, d), (c, d).
                # a = i*m_sub + j
                # b = j*p_sub + k
                # c = k*n_sub + i
                # So R_sub[a, b, c] should be 1.
                # Let's check the specific entries.
                max_diff = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            a = i * m_sub + j
                            b = j * p_sub + k
                            c = k * n_sub + i
                            max_diff = max(max_diff, abs(R_sub[a, b, c] - 1.0))
                
                if max_diff < 1e-6:
                    converged = True
                    break
            
            if converged:
                return U, V, W, r
        
        return None, None, None, None

    # Attempt to solve sub-problem
    if time.time() < deadline - 10:
        sub_U, sub_V, sub_W, sub_rank = solve_sub_problem(max_time=40.0, target_rank=15)
        
        if sub_rank is not None and sub_rank < n_sub * m_sub * p_sub:
            total_rank = 4 * sub_rank
            if total_rank < best_rank:
                # Construct full U, V, W
                U_full = np.zeros((nm, total_rank))
                V_full = np.zeros((mp, total_rank))
                W_full = np.zeros((pn, total_rank))
                
                for block_i in range(2): # A1, A2
                    for block_k in range(2): # B1, B2
                        offset = (block_i * 2 + block_k) * sub_rank
                        
                        for s_i in range(n_sub): # 0,1
                            full_i = 2 * block_i + s_i
                            for j in range(m_sub): # 0..4
                                full_j = j
                                for s_k in range(p_sub): # 0..2
                                    full_k = 3 * block_k + s_k
                                    
                                    # U index
                                    a_sub = s_i * m_sub + j
                                    a_full = full_i * m + full_j
                                    U_full[a_full, offset:offset+sub_rank] += sub_U[a_sub, :]
                                    
                                    # V index
                                    b_sub = j * p_sub + s_k
                                    b_full = full_j * p + full_k
                                    V_full[b_full, offset:offset+sub_rank] += sub_V[b_sub, :]
                                    
                                    # W index
                                    c_sub = s_k * n_sub + s_i
                                    c_full = full_k * n + full_i
                                    W_full[c_full, offset:offset+sub_rank] += sub_W[c_sub, :]
                
                # Verify full solution
                if verify_fast(U_full, V_full, W_full):
                    best_U, best_V, best_W = U_full, V_full, W_full
                    best_rank = total_rank

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
