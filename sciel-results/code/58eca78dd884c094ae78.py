import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Helper to check if a decomposition is valid
    def check_validity_residual(U, V, W, tol=1e-6):
        R = U.shape[1]
        err = 0.0
        V_mat = V.reshape(m, p, R) # V_mat[j, k, r]
        U_mat = U.reshape(n, m, R) # U_mat[i, j, r]
        W_mat = W.reshape(p, n, R) # W_mat[k, i, r]
        
        for i in range(n):
            if time.time() > deadline:
                return float('inf')
            for k in range(p):
                # U_i is (m, R), V_k is (m, R), W_k_i is (R,)
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                W_k_i = W_mat[k, i, :]
                
                # M_ik should be (m, m) identity
                # M_ik[j, l] = sum_r U_i[j, r] * W_k_i[r] * V_k[l, r]
                # M_ik = U_i @ diag(W_k_i) @ V_k.T
                M_ik = U_i @ np.diag(W_k_i) @ V_k.T
                
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err

    def get_rank_55_solution():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # Part A: j=0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
            
        # Part B: j=1,2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
            
        return U, V, W

    U, V, W = get_rank_55_solution()
    
    # Verify validity
    err = check_validity_residual(U, V, W)
    if err > 1e-6:
        # Fallback to schoolbook if construction failed (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
