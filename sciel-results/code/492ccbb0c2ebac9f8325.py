import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Step 1: Find decomposition for M_{2,3,3}
    # Target tensor T_{i,j,k} = 1 for i in 0..1, j in 0..2, k in 0..2
    n_small, m_small, p_small = 2, 3, 3
    T_target = np.ones((n_small, m_small, p_small))
    
    best_rank_small = n_small * m_small * p_small # 18
    best_U_small = None
    best_V_small = None
    best_W_small = None
    
    # Try ranks from 13 up to 18
    for R in range(13, best_rank_small + 1):
        if time.time() > deadline - 20: break
        
        found = False
        for restart in range(10):
            if time.time() > deadline - 20: break
            
            # Initialize
            U = np.random.randn(n_small * m_small, R) * 0.1
            V = np.random.randn(m_small * p_small, R) * 0.1
            W = np.random.randn(p_small * n_small, R) * 0.1
            
            def error(params):
                U_flat = params[:n_small*m_small*R].reshape(n_small*m_small, R)
                V_flat = params[n_small*m_small*R:n_small*m_small*R + m_small*p_small*R].reshape(m_small*p_small, R)
                W_flat = params[n_small*m_small*R + m_small*p_small*R:].reshape(p_small*n_small, R)
                
                # Compute reconstruction
                # T_recon[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
                # Efficient computation:
                # U is (n*m, R), V is (m*p, R), W is (p*n, R)
                # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
                U_3d = U_flat.reshape(n_small, m_small, R)
                V_3d = V_flat.reshape(m_small, p_small, R)
                W_3d = W_flat.reshape(p_small, n_small, R)
                
                # Sum over r: U[i,j,r] V[j,k,r] W[i,k,r]
                # This is a tensor contraction.
                # tmp[i,j,k] = sum_r U[i,j,r] V[j,k,r]
                # tmp[i,j,k] = np.einsum('ijr,jkr->ijk', U_3d, V_3d)
                # result[i,k] = sum_j tmp[i,j,k] W[i,k,r]? No.
                # The term is U[i,j,r] * V[j,k,r] * W[i,k,r].
                # Sum over r.
                # result[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
                # We can do:
                # prod = U_3d * V_3d * W_3d? No, dimensions don't align directly for element-wise.
                # U[i,j,r], V[j,k,r], W[i,k,r].
                # We need to align j.
                # Let's use einsum: 'ijr,jkr,ikr->ijk'
                try:
                    recon = np.einsum('ijr,jkr,ikr->ijk', U_3d, V_3d, W_3d)
                except:
                    return 1e10
                
                err = np.sum((recon - T_target)**2)
                return err
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B
            res = opt.minimize(error, params0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.fun < 1e-9:
                # Check if we can round to half-integers
                U_res = res.x[:n_small*m_small*R].reshape(n_small*m_small, R)
                V_res = res.x[n_small*m_small*R:n_small*m_small*R + m_small*p_small*R].reshape(m_small*p_small, R)
                W_res = res.x[n_small*m_small*R + m_small*p_small*R:].reshape(p_small*n_small, R)
                
                # Try rounding to nearest 0.5
                U_round = np.round(U_res * 2) / 2
                V_round = np.round(V_res * 2) / 2
                W_round = np.round(W_res * 2) / 2
                
                # Verify exactness with rounded values
                U_3d = U_round.reshape(n_small, m_small, R)
                V_3d = V_round.reshape(m_small, p_small, R)
                W_3d = W_round.reshape(p_small, n_small, R)
                recon = np.einsum('ijr,jkr,ikr->ijk', U_3d, V_3d, W_3d)
                
                if np.allclose(recon, T_target, atol=1e-9):
                    best_rank_small = R
                    best_U_small = U_round
                    best_V_small = V_round
                    best_W_small = W_round
                    found = True
                    break
        
        if found:
            break
    
    # Step 2: Construct M_{2,3,9} decomposition
    if best_U_small is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    R_small = best_rank_small
    R_total = 3 * R_small
    
    U = [[0] * R_total for _ in range(n * m)]
    V = [[0] * R_total for _ in range(m * p)]
    W = [[0] * R_total for _ in range(p * n)]
    
    # best_U_small is shape (n_small*m_small, R_small)
    # n=2, m=3, p=9
    # n_small=2, m_small=3, p_small=3
    
    for t in range(3):
        r_offset = t * R_small
        for i in range(n_small):
            for j in range(m_small):
                # U index: i*m + j
                idx_u = i * m + j
                for r in range(R_small):
                    U[idx_u][r_offset + r] = best_U_small[idx_u][r]
        
        for j in range(m_small):
            for k_prime in range(p_small):
                k = t * p_small + k_prime
                idx_v = j * p + k
                for r in range(R_small):
                    V[idx_v][r_offset + r] = best_V_small[j * p_small + k_prime][r]
        
        for k_prime in range(p_small):
            for i in range(n_small):
                k = t * p_small + k_prime
                idx_w = k * n + i
                for r in range(R_small):
                    W[idx_w][r_offset + r] = best_W_small[k_prime * n_small + i][r]
    
    # Convert to lists of lists if necessary, though numpy arrays are often accepted
    # The prompt example uses lists of lists.
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
