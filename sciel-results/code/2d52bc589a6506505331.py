"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    deadline = time.time() + 140
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = n*m*p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try alternating minimization for lower ranks
    for target_rank in range(n*m*p - 1, 20, -1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_r = np.random.randn(nm, target_rank) * 0.1
            V_r = np.random.randn(mp, target_rank) * 0.1
            W_r = np.random.randn(pn, target_rank) * 0.1
            
            # Alternating minimization
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for U given V and W
                for i in range(n):
                    for j in range(m):
                        # System: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for k=0..p-1
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            A_mat[k, :] = V_r[j*p+k, :] * W_r[k*n+i, :]
                        b_vec = np.ones(p)
                        # Solve A_mat @ U[i*m+j, :] = b_vec
                        try:
                            U_r[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        except:
                            pass
                
                # Solve for V given U and W
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            A_mat[i, :] = U_r[i*m+j, :] * W_r[k*n+i, :]
                        b_vec = np.ones(n)
                        try:
                            V_r[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        except:
                            pass
                
                # Solve for W given U and V
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            A_mat[j, :] = U_r[i*m+j, :] * V_r[j*p+k, :]
                        b_vec = np.ones(m)
                        try:
                            W_r[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        except:
                            pass
                
                # Check residual
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i*m + j
                            idx_v = j*p + k
                            idx_w = k*n + i
                            val = np.sum(U_r[idx_u, :] * V_r[idx_v, :] * W_r[idx_w, :])
                            residual += (val - 1)**2
                
                if residual < 1e-10:
                    # Try to round
                    U_round = np.round(U_r * 2) / 2
                    V_round = np.round(V_r * 2) / 2
                    W_round = np.round(W_r * 2) / 2
                    
                    # Verify
                    residual_round = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i*m + j
                                idx_v = j*p + k
                                idx_w = k*n + i
                                val = np.sum(U_round[idx_u, :] * V_round[idx_v, :] * W_round[idx_w, :])
                                residual_round += (val - 1)**2
                    
                    if residual_round < 1e-9:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_round, V_round, W_round
                        found = True
                        break
            
            if found:
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
