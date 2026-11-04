import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # The problem reduces to 3 copies of 4x4x3
    # We search for rank r for 4x4x3
    # Schoolbook rank for 4x4x3 is 48
    # Total schoolbook rank is 144
    
    # Try to find a decomposition for 4x4x3 with rank < 48
    # If found, replicate 3 times
    
    n_sub, m_sub, p_sub = 4, 4, 3
    rank_sub_schoolbook = n_sub * m_sub * p_sub # 48
    
    best_rank = rank_sub_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook decomposition for 4x4x3
    U_sb = np.zeros((n_sub * m_sub, rank_sub_schoolbook))
    V_sb = np.zeros((m_sub * p_sub, rank_sub_schoolbook))
    W_sb = np.zeros((p_sub * n_sub, rank_sub_schoolbook))
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U_sb[i * m_sub + j, col] = 1
                V_sb[j * p_sub + k, col] = 1
                W_sb[k * n_sub + i, col] = 1
                col += 1
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to improve
    # Use a simple optimization for r = 47
    # Time limit check
    start_time = time.time()
    time_limit = 140 # Leave 10s for output
    
    def loss(U, V, W):
        # Compute sum_a U[i,j,a] V[j,k,a] W[k,i,a] - 1
        # Vectorized:
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_r = U.reshape(n_sub, m_sub, -1)
        V_r = V.reshape(m_sub, p_sub, -1)
        W_r = W.reshape(p_sub, n_sub, -1)
        # Sum over a
        # Result shape (n, m, p)
        # T[i,j,k] = sum_a U_r[i,j,a] V_r[j,k,a] W_r[k,i,a]
        # This is a tensor contraction.
        # Use np.einsum
        T = np.einsum('ija,jka,ki->ijk', U_r, V_r, W_r)
        # Target is 1 everywhere
        return np.sum((T - 1.0)**2)
        
    def gradients(U, V, W):
        U_r = U.reshape(n_sub, m_sub, -1)
        V_r = V.reshape(m_sub, p_sub, -1)
        W_r = W.reshape(p_sub, n_sub, -1)
        T = np.einsum('ija,jka,ki->ijk', U_r, V_r, W_r)
        E = T - 1.0
        # Gradient wrt U_r[i,j,a] = 2 * sum_{k} E[i,j,k] * V_r[j,k,a] * W_r[k,i,a]
        # grad_U_r = 2 * np.einsum('ijk,jka,ki->ija', E, V_r, W_r)
        # Gradient wrt V_r[j,k,a] = 2 * sum_{i} E[i,j,k] * U_r[i,j,a] * W_r[k,i,a]
        # grad_V_r = 2 * np.einsum('ijk,ija,ki->jka', E, U_r, W_r)
        # Gradient wrt W_r[k,i,a] = 2 * sum_{j} E[i,j,k] * U_r[i,j,a] * V_r[j,k,a]
        # grad_W_r = 2 * np.einsum('ijk,ija,jka->ki', E, U_r, V_r) -- Wait, output shape should be (p,n,r) i.e., (k,i,a)
        # grad_W_r = 2 * np.einsum('ijk,ija,jka->kia', E, U_r, V_r)
        
        gU = 2.0 * np.einsum('ijk,jka,ki->ija', E, V_r, W_r)
        gV = 2.0 * np.einsum('ijk,ija,ki->jka', E, U_r, W_r)
        gW = 2.0 * np.einsum('ijk,ija,jka->kia', E, U_r, V_r)
        
        return gU.reshape(n_sub * m_sub, -1), gV.reshape(m_sub * p_sub, -1), gW.reshape(p_sub * n_sub, -1)

    # Try r = 47
    r_try = 47
    if time.time() - start_time < time_limit * 0.5:
        # Use Adam optimizer
        def adam_step(params, grads, m_params, v_params, t, lr, b1, b2, eps):
            for i in range(len(params)):
                m_params[i] = b1 * m_params[i] + (1 - b1) * grads[i]
                v_params[i] = b2 * v_params[i] + (1 - b2) * (grads[i] ** 2)
                m_hat = m_params[i] / (1 - b1 ** t)
                v_hat = v_params[i] / (1 - b2 ** t)
                params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        for restart in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            U_try = np.random.randn(n_sub * m_sub, r_try) * 0.1
            V_try = np.random.randn(m_sub * p_sub, r_try) * 0.1
            W_try = np.random.randn(p_sub * n_sub, r_try) * 0.1
            
            params = [U_try, V_try, W_try]
            m_params = [np.zeros_like(p) for p in params]
            v_params = [np.zeros_like(p) for p in params]
            
            lr = 0.01
            b1 = 0.9
            b2 = 0.999
            eps = 1e-8
            
            for step in range(2000):
                if time.time() - start_time > time_limit - 10:
                    break
                
                loss_val = loss(U_try, V_try, W_try)
                if loss_val < 1e-6:
                    best_U = U_try
                    best_V = V_try
                    best_W = W_try
                    best_rank = r_try
                    break
                
                gU, gV, gW = gradients(U_try, V_try, W_try)
                grads = [gU, gV, gW]
                
                adam_step(params, grads, m_params, v_params, step + 1, lr, b1, b2, eps)
                
                U_try = params[0]
                V_try = params[1]
                W_try = params[2]
            
            if best_rank == r_try:
                break

    # Construct final U, V, W for 4x4x9
    # Replicate the sub-decomposition 3 times
    rank_total = best_rank * 3
    U_final = np.zeros((n * m, rank_total))
    V_final = np.zeros((m * p, rank_total))
    W_final = np.zeros((p * n, rank_total))
    
    # Map indices
    # k in 0..8
    # Block 0: k=0,1,2 -> use best_U, best_V, best_W
    # Block 1: k=3,4,5 -> use best_U, best_V, best_W
    # Block 2: k=6,7,8 -> use best_U, best_V, best_W
    
    for block in range(3):
        offset = block * best_rank
        # U is same for all blocks (depends on i,j)
        U_final[:, offset:offset+best_rank] = best_U
        
        # V depends on j,k
        # k_local = k % 3
        # k_global = block * 3 + k_local
        for j in range(m):
            for k_local in range(p_sub):
                k_global = block * p_sub + k_local
                idx_v = j * p + k_global
                idx_v_sub = j * p_sub + k_local
                V_final[idx_v, offset:offset+best_rank] = best_V[idx_v_sub]
                
        # W depends on k,i
        for k_local in range(p_sub):
            k_global = block * p_sub + k_local
            for i in range(n):
                idx_w = k_global * n + i
                idx_w_sub = k_local * n_sub + i
                W_final[idx_w, offset:offset+best_rank] = best_W[idx_w_sub]
                
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
