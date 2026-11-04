import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 144
    
    # Generate schoolbook decomposition
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_schoolbook

    # Search for better rank
    # Try to find a decomposition for 4x4x3 with rank r_sub < 48
    # Total rank = 3 * r_sub
    # Target r_sub = 45 (Total 135)
    
    n_sub, m_sub, p_sub = 4, 4, 3
    start_time = time.time()
    time_limit = 140
    
    def verify_sub(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check if sum_a U[i,j,a] V[j,k,a] W[k,i,a] == 1 for all i,j,k
        U_r = U.reshape(n_sub, m_sub, -1)
        V_r = V.reshape(m_sub, p_sub, -1)
        W_r = W.reshape(p_sub, n_sub, -1)
        T = np.einsum('ija,jka,ki->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)

    def is_half_integer(arr):
        # Check if all entries are multiples of 0.5
        return np.allclose(arr * 2, np.round(arr * 2), atol=1e-9)

    # Try r_sub = 45
    r_sub_target = 45
    if time.time() - start_time < time_limit * 0.8:
        for restart in range(500):
            if time.time() - start_time > time_limit:
                break
            
            # Random binary U and V
            U_try = np.random.randint(0, 2, (n_sub * m_sub, r_sub_target))
            V_try = np.random.randint(0, 2, (m_sub * p_sub, r_sub_target))
            
            # Construct linear system for W
            # Variables: W_{k,i,a} for k in 0..2, i in 0..3, a in 0..r-1
            # Total variables: 3 * 4 * r_sub_target = 12 * r_sub_target
            num_vars = p_sub * n_sub * r_sub_target
            num_eqs = n_sub * m_sub * p_sub # 48
            
            A_mat = np.zeros((num_eqs, num_vars))
            b_vec = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        # Equation: sum_a U[i,j,a] V[j,k,a] W[k,i,a] = 1
                        # Variable index for W[k,i,a] is (k * n_sub + i) * r_sub_target + a
                        for a in range(r_sub_target):
                            coeff = U_try[i * m_sub + j, a] * V_try[j * p_sub + k, a]
                            if coeff != 0:
                                var_idx = (k * n_sub + i) * r_sub_target + a
                                A_mat[eq_idx, var_idx] = coeff
                        eq_idx += 1
            
            # Solve A_mat @ W_flat = b_vec
            # Use least squares since it's underdetermined
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
                
            W_try = W_flat.reshape(p_sub * n_sub, r_sub_target)
            
            # Check if W_try is half-integer
            if is_half_integer(W_try):
                if verify_sub(U_try, V_try, W_try):
                    # Found a valid decomposition for 4x4x3 with rank 45
                    # Construct full decomposition
                    best_rank = r_sub_target * 3
                    U_final = np.zeros((n * m, best_rank))
                    V_final = np.zeros((m * p, best_rank))
                    W_final = np.zeros((p * n, best_rank))
                    
                    for block in range(3):
                        offset = block * r_sub_target
                        U_final[:, offset:offset+r_sub_target] = U_try
                        for j in range(m):
                            for k_local in range(p_sub):
                                k_global = block * p_sub + k_local
                                idx_v = j * p + k_global
                                idx_v_sub = j * p_sub + k_local
                                V_final[idx_v, offset:offset+r_sub_target] = V_try[idx_v_sub]
                        for k_local in range(p_sub):
                            k_global = block * p_sub + k_local
                            for i in range(n):
                                idx_w = k_global * n + i
                                idx_w_sub = k_local * n_sub + i
                                W_final[idx_w, offset:offset+r_sub_target] = W_try[idx_w_sub]
                    
                    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    # Return schoolbook if no better found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
