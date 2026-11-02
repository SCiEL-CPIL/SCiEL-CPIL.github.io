import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find a better decomposition for 2x3x2 and replicate
    n_sub, m_sub, p_sub = 2, 3, 2
    target_rank_sub = 11 # Try rank 11 first
    best_sub = None
    
    start_time = time.time()
    time_limit = 100.0
    
    for R in range(target_rank_sub, 12):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        U = np.random.randn(n_sub * m_sub, R)
        V = np.random.randn(m_sub * p_sub, R)
        W = np.random.randn(p_sub * n_sub, R)
        
        # ALS Iteration
        for it in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Solve for U
            # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
            # For fixed V, W, U is linear.
            # Minimize sum_{b,c} (sum_r U_{a,r} (V_{b,r} W_{c,r}) - T_{a,b,c})^2
            # This is a linear system: M_U U_a = rhs_a
            # M_U = sum_{b,c} (V_{b,:} * W_{c,:})^T (V_{b,:} * W_{c,:})
            # rhs_a = sum_{b,c} T_{a,b,c} (V_{b,:} * W_{c,:})
            
            M_U = np.zeros((R, R))
            rhs_U = np.zeros((n_sub * m_sub, R))
            
            for a in range(n_sub * m_sub):
                for b in range(m_sub * p_sub):
                    for c in range(p_sub * n_sub):
                        T_val = 1.0 if (a // m_sub == c // n_sub and a % m_sub == b // p_sub and b % p_sub == c % n_sub) else 0.0
                        # Note: T_{a,b,c} is 1 if indices match multiplication pattern
                        # a = i*m + j, b = j*p + k, c = k*n + i
                        # Match if i_a == i_c, j_a == j_b, k_b == k_c
                        i_a, j_a = divmod(a, m_sub)
                        j_b, k_b = divmod(b, p_sub)
                        k_c, i_c = divmod(c, n_sub)
                        if i_a == i_c and j_a == j_b and k_b == k_c:
                            T_val = 1.0
                        else:
                            T_val = 0.0
                        
                        v_w = V[b, :] * W[c, :]
                        M_U += np.outer(v_w, v_w)
                        if T_val != 0:
                            rhs_U[a] += T_val * v_w
            
            M_U += 1e-6 * np.eye(R) # Regularization
            try:
                U = np.linalg.solve(M_U.T, rhs_U.T).T
            except:
                U = np.linalg.lstsq(M_U.T, rhs_U.T, rcond=None)[0].T
            
            # Solve for V
            M_V = np.zeros((R, R))
            rhs_V = np.zeros((m_sub * p_sub, R))
            for b in range(m_sub * p_sub):
                for a in range(n_sub * m_sub):
                    for c in range(p_sub * n_sub):
                        i_a, j_a = divmod(a, m_sub)
                        j_b, k_b = divmod(b, p_sub)
                        k_c, i_c = divmod(c, n_sub)
                        if i_a == i_c and j_a == j_b and k_b == k_c:
                            T_val = 1.0
                        else:
                            T_val = 0.0
                        
                        u_w = U[a, :] * W[c, :]
                        M_V += np.outer(u_w, u_w)
                        if T_val != 0:
                            rhs_V[b] += T_val * u_w
            
            M_V += 1e-6 * np.eye(R)
            try:
                V = np.linalg.solve(M_V.T, rhs_V.T).T
            except:
                V = np.linalg.lstsq(M_V.T, rhs_V.T, rcond=None)[0].T
            
            # Solve for W
            M_W = np.zeros((R, R))
            rhs_W = np.zeros((p_sub * n_sub, R))
            for c in range(p_sub * n_sub):
                for a in range(n_sub * m_sub):
                    for b in range(m_sub * p_sub):
                        i_a, j_a = divmod(a, m_sub)
                        j_b, k_b = divmod(b, p_sub)
                        k_c, i_c = divmod(c, n_sub)
                        if i_a == i_c and j_a == j_b and k_b == k_c:
                            T_val = 1.0
                        else:
                            T_val = 0.0
                        
                        u_v = U[a, :] * V[b, :]
                        M_W += np.outer(u_v, u_v)
                        if T_val != 0:
                            rhs_W[c] += T_val * u_v
            
            M_W += 1e-6 * np.eye(R)
            try:
                W = np.linalg.solve(M_W.T, rhs_W.T).T
            except:
                W = np.linalg.lstsq(M_W.T, rhs_W.T, rcond=None)[0].T
            
            # Check Error
            err = 0.0
            for a in range(n_sub * m_sub):
                for b in range(m_sub * p_sub):
                    for c in range(p_sub * n_sub):
                        i_a, j_a = divmod(a, m_sub)
                        j_b, k_b = divmod(b, p_sub)
                        k_c, i_c = divmod(c, n_sub)
                        target = 1.0 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0.0
                        val = np.dot(U[a, :], np.dot(V[b, :], W[c, :]))
                        err += (val - target) ** 2
                        
            if err < 1e-10:
                best_sub = (U, V, W, R)
                break
        
        if best_sub:
            break

    if best_sub:
        U_sub, V_sub, W_sub, R_sub = best_sub
        # Snap to half-integers
        U_sub = np.round(U_sub * 2) / 2
        V_sub = np.round(V_sub * 2) / 2
        W_sub = np.round(W_sub * 2) / 2
        
        # Verify again
        err = 0.0
        for a in range(n_sub * m_sub):
            for b in range(m_sub * p_sub):
                for c in range(p_sub * n_sub):
                    i_a, j_a = divmod(a, m_sub)
                    j_b, k_b = divmod(b, p_sub)
                    k_c, i_c = divmod(c, n_sub)
                    target = 1.0 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0.0
                    val = np.dot(U_sub[a, :], np.dot(V_sub[b, :], W_sub[c, :]))
                    err += (val - target) ** 2
        
        if err < 1e-9:
            # Construct full 2x3x10
            num_blocks = p // p_sub # 5
            rank_full = R_sub * num_blocks
            U_full = [[0.0] * rank_full for _ in range(n * m)]
            V_full = [[0.0] * rank_full for _ in range(m * p)]
            W_full = [[0.0] * rank_full for _ in range(p * n)]
            
            for block in range(num_blocks):
                offset = block * R_sub
                # U is same for all blocks?
                # U indices i, j. i in 0..1, j in 0..2.
                # In sub-problem, i in 0..1, j in 0..2.
                # So U_full[i*m+j] = U_sub[i*m_sub+j]
                for i in range(n):
                    for j in range(m):
                        U_full[i * m + j][offset : offset + R_sub] = U_sub[i * m_sub + j].tolist()
                
                # V indices j, k.
                # In sub-problem, k in 0..1.
                # In full problem, k = block * 2 + k_sub.
                for j in range(m):
                    for k_sub in range(p_sub):
                        k_full = block * p_sub + k_sub
                        V_full[j * p + k_full][offset : offset + R_sub] = V_sub[j * m_sub + k_sub].tolist()
                        
                # W indices k, i.
                # In sub-problem, k in 0..1.
                # In full problem, k = block * 2 + k_sub.
                for k_sub in range(p_sub):
                    k_full = block * p_sub + k_sub
                    for i in range(n):
                        W_full[k_full * n + i][offset : offset + R_sub] = W_sub[k_sub * n_sub + i].tolist()
                        
            return {"U": U_full, "V": V_full, "W": W_full}
            
    return fallback

