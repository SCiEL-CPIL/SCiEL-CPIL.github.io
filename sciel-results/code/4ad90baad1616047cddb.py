import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 3, 6
    
    # Build target tensor T for 2x3x6
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Strategy: Use two independent 2x3x3 blocks
    # Block 1: k in {0,1,2}, Block 2: k in {3,4,5}
    # Each 2x3x3 has rank 15, so total rank = 30
    
    # For 2x3x3, construct explicit rank-15 decomposition
    n2, m2, p2 = 2, 3, 3
    
    # The 2x3x3 multiplication tensor T233[i*3+j, j*3+k, k*2+i] = 1
    # We need U233 (6x15), V233 (9x15), W233 (6x15)
    
    # Use a structured approach: exploit the fact that for 2x3x3,
    # we can use the "slice" structure
    
    # Known explicit rank-15 decomposition for 2x3x3:
    # The 18 multiplications x_i * y_j * z_k can be grouped
    
    # Let's use a direct linear algebra approach with rational arithmetic
    # via numpy, then round to nearest half-integer
    
    def solve_233_rank15():
        """Find rank-15 decomposition of 2x3x3 multiplication tensor"""
        R = 15
        rows_u = 6
        rows_v = 9
        rows_w = 6
        
        T233 = np.zeros((rows_u, rows_v, rows_w))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
        # Use ALS with multiple restarts, trying to get exact solutions
        best = None
        
        for seed in range(200):
            if time.time() > deadline - 20:
                break
                
            rng = np.random.RandomState(seed)
            U = rng.randn(rows_u, R) * 0.5
            V = rng.randn(rows_v, R) * 0.5
            W = rng.randn(rows_w, R) * 0.5
            
            for it in range(150):
                # Solve for U given V, W
                Z_vw = np.zeros((R, rows_v * rows_w))
                for r in range(R):
                    Z_vw[r, :] = np.outer(V[:, r], W[:, r]).flatten()
                T_flat_u = T233.reshape(rows_u, -1).T
                U_sol, _, _, _ = np.linalg.lstsq(Z_vw.T, T_flat_u, rcond=None)
                U = U_sol.T
                
                # Solve for V given U, W
                Z_uw = np.zeros((R, rows_u * rows_w))
                for r in range(R):
                    Z_uw[r, :] = np.outer(U[:, r], W[:, r]).flatten()
                T_flat_v = T233.transpose(1, 0, 2).reshape(rows_v, -1).T
                V_sol, _, _, _ = np.linalg.lstsq(Z_uw.T, T_flat_v, rcond=None)
                V = V_sol.T
                
                # Solve for W given U, V
                Z_uv = np.zeros((R, rows_u * rows_v))
                for r in range(R):
                    Z_uv[r, :] = np.outer(U[:, r], V[:, r]).flatten()
                T_flat_w = T233.transpose(2, 0, 1).reshape(rows_w, -1).T
                W_sol, _, _, _ = np.linalg.lstsq(Z_uv.T, T_flat_w, rcond=None)
                W = W_sol.T
            
            # Check if solution is close to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_check = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.allclose(T_check, T233, atol=1e-9):
                return (U_r, V_r, W_r)
            
            # Try integer rounding
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            T_check_i = np.einsum('ir,jr,kr->ijk', U_i, V_i, W_i)
            if np.allclose(T_check_i, T233, atol=1e-9):
                return (U_i, V_i, W_i)
        
        return None
    
    result_233 = solve_233_rank15()
    
    if result_233 is not None:
        U233, V233, W233 = result_233
        R233 = 15
        R236 = 2 * R233
        
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        # Embed block 1 (k=0,1,2) and block 2 (k=3,4,5)
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    u_idx = i * m2 + j
                    v_idx_233 = j * p2 + k
                    w_idx_233 = k * n2 + i
                    
                    # Block 1
                    v_idx_b1 = j * p + k
                    w_idx_b1 = k * n2 + i
                    U236[u_idx, :R233] = U233[u_idx, :]
                    V236[v_idx_b1, :R233] = V233[v_idx_233, :]
                    W236[w_idx_b1, :R233] = W233[w_idx_233, :]
                    
                    # Block 2
                    v_idx_b2 = j * p + (k + p2)
                    w_idx_b2 = (k + p2) * n2 + i
                    U236[u_idx, R233:] = U233[u_idx, :]
                    V236[v_idx_b2, R233:] = V233[v_idx_233, :]
                    W236[w_idx_b2, R233:] = W233[w_idx_233, :]
        
        T_final = np.einsum('ir,jr,kr->ijk', U236, V236, W236)
        if np.allclose(T_final, T, atol=1e-9):
            return {
                "U": U236.tolist(),
                "V": V236.tolist(),
                "W": W236.tolist()
            }
    
    # Fallback: schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
