import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0
    
    # 1. Try to find a low-rank decomposition of the 2x3x3 block
    # M_{2,3,3} tensor
    n2, m2, p2 = 2, 3, 3
    T_233 = np.zeros((n2 * m2, m2 * p2, p2 * n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T_233[i * m2 + j, j * p2 + k, k * n2 + i] = 1.0
                
    T_U_233 = T_233.reshape(n2 * m2, m2 * p2 * p2 * n2)
    T_V_233 = T_233.transpose(1, 0, 2).reshape(m2 * p2, n2 * m2 * p2 * n2)
    T_W_233 = T_233.transpose(2, 0, 1).reshape(p2 * n2, n2 * m2 * m2 * p2)
    
    def solve_factor(T_mat, other1, other2):
        dim2 = other1.shape[0]
        dim3 = other2.shape[0]
        rank = other1.shape[1]
        M = (other1.reshape(dim2, 1, rank) * other2.reshape(1, dim3, rank)).reshape(dim2 * dim3, rank)
        try:
            factor_T, residuals, rank_2, sv = np.linalg.lstsq(M.T, T_mat.T, rcond=None)
            return factor_T.T
        except:
            return T_mat @ np.linalg.pinv(M.T)
            
    def als_solve(T_U, T_V, T_W, U, V, W, max_iter=200, tol=1e-10):
        for it in range(max_iter):
            U_new = solve_factor(T_U, V, W)
            V_new = solve_factor(T_V, U_new, W)
            W_new = solve_factor(T_W, U_new, V_new)
            
            delta_U = np.linalg.norm(U_new - U)
            delta_V = np.linalg.norm(V_new - V)
            delta_W = np.linalg.norm(W_new - W)
            
            U, V, W = U_new, V_new, W_new
            
            if delta_U + delta_V + delta_W < tol:
                break
        return U, V, W
        
    def check_decomposition(U, V, W, T):
        T_rec = np.zeros_like(T)
        for c in range(U.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        max_err = np.max(np.abs(T_rec - T))
        return max_err, max_err < 1e-8

    # Search for 2x3x3 decomposition
    best_233 = None
    best_233_rank = n2 * m2 * p2 # 18
    
    # Try ranks for 2x3x3
    target_ranks_233 = [7, 8, 9, 10, 12, 14, 16, 18]
    
    for r in target_ranks_233:
        if time.time() > deadline - 30: # Leave 30s for main construction
            break
            
        found = False
        n_attempts = 20 if r <= 9 else 10
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break
                
            np.random.seed(hash((r, attempt)) % (2**32))
            U = np.random.randn(n2 * m2, r)
            V = np.random.randn(m2 * p2, r)
            W = np.random.randn(p2 * n2, r)
            
            U, V, W = als_solve(T_U_233, T_V_233, T_W_233, U, V, W, max_iter=50)
            
            max_err, valid = check_decomposition(U, V, W, T_233)
            
            if valid:
                best_233 = (U, V, W)
                best_233_rank = r
                found = True
                break
                
        if found:
            break
            
    # Construct 2x3x9 decomposition from 2x3x3
    if best_233 is not None:
        U_233, V_233, W_233 = best_233
        r_block = best_233_rank
        r_total = r_block * 3
        
        U_total = np.zeros((n * m, r_total))
        V_total = np.zeros((m * p, r_total))
        W_total = np.zeros((p * n, r_total))
        
        for q in range(3):
            col_start = q * r_block
            col_end = (q + 1) * r_block
            
            # U is repeated
            U_total[:, col_start:col_end] = U_233
            
            # V mapping:
            # V_total index: j * 9 + 3q + k'
            # V_233 index: j * 3 + k'
            for j in range(3):
                for k_prime in range(3):
                    v_idx = j * 9 + 3 * q + k_prime
                    v0_idx = j * 3 + k_prime
                    V_total[v_idx, col_start:col_end] = V_233[v0_idx, :]
                    
            # W mapping:
            # W_total index: (3q + k') * 2 + i
            # W_233 index: k' * 2 + i
            for k_prime in range(3):
                for i in range(2):
                    w_idx = (3 * q + k_prime) * 2 + i
                    w0_idx = k_prime * 2 + i
                    W_total[w_idx, col_start:col_end] = W_233[w0_idx, :]
                    
        # Verify total decomposition
        # Build T_239
        T_239 = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_239[i * m + j, j * p + k, k * n + i] = 1.0
                    
        T_rec = np.zeros_like(T_239)
        for c in range(r_total):
            T_rec += np.einsum('i,j,k->ijk', U_total[:, c], V_total[:, c], W_total[:, c])
            
        max_err = np.max(np.abs(T_rec - T_239))
        
        if max_err < 1e-8:
            return {
                "U": U_total.tolist(),
                "V": V_total.tolist(),
                "W": W_total.tolist()
            }
            
    # Fallback: Schoolbook
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
                
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
