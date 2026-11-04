import numpy as np
import time

def construct():
    start_time = time.time()
    max_time = 140
    
    # Target: 2x3x4
    # Strategy: Find rank 10 decomposition for 2x3x2, then use two copies for 2x3x4
    n, m, p = 2, 3, 2
    target_rank = 10
    
    # Build target tensor for 2x3x2
    T = np.ones((n, m, p))
    
    def try_rank_232(R, max_restarts=100):
        """Try to find rank R decomposition for 2x3x2."""
        def unpack(params):
            U = params[:R * n * m].reshape(R, n * m).T
            V = params[R * n * m : R * (n * m + m * p)].reshape(R, m * p).T
            W = params[R * (n * m + m * p) :].reshape(R, p * n).T
            return U, V, W
        def residuals(params):
            U, V, W = unpack(params)
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            return (T - A_approx).flatten()
            
        num_params = R * (n * m + m * p + p * n)
        
        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                return None
            
            init_params = np.random.uniform(-1, 1, size=num_params)
            try:
                from scipy.optimize import least_squares
                # Use 'lm' for small problems? 'trf' is safer.
                sol = least_squares(residuals, init_params, method='trf', max_nfev=200)
                
                if sol.cost < 1e-6:
                    # Snap to integers
                    rounded_params = np.round(sol.x)
                    res_check = residuals(rounded_params)
                    if np.linalg.norm(res_check) < 1e-9:
                        U, V, W = unpack(rounded_params)
                        return {"U": U, "V": V, "W": W}
            except Exception:
                pass
        return None

    # Attempt to find rank 10 for 2x3x2
    dec_232 = try_rank_232(10)
    
    if dec_232:
        # Construct 2x3x4 from two copies
        # 2x3x4: n=2, m=3, p=4
        # U_234 (6 x 20)
        # V_234 (12 x 20)
        # W_234 (8 x 20)
        
        U_232 = dec_232["U"] # 6 x 10
        V_232 = dec_232["V"] # 6 x 10 (j=3, p=2 -> 6 entries)
        W_232 = dec_232["W"] # 4 x 10 (k=2, i=2 -> 4 entries)
        
        U_234 = np.zeros((6, 20))
        V_234 = np.zeros((12, 20))
        W_234 = np.zeros((8, 20))
        
        # Copy 1: For k in {0, 1} (corresponds to p=2 block)
        # U rows: i*3+j (same)
        # V rows: j*4+k. For block 1, k is 0,1. So V_234[0:6] = V_232
        # W rows: k*2+i. For block 1, k=0,1. So W_234[0:4] = W_232
        U_234[:, 0:10] = U_232
        V_234[0:6, 0:10] = V_232
        W_234[0:4, 0:10] = W_232
        
        # Copy 2: For k in {2, 3}
        # U is same, so duplicate U_232
        # V_234 rows: j*4+k. For block 2, k=2,3.
        # j=0 -> k=2 (row 2), k=3 (row 3)
        # j=1 -> k=2 (row 6), k=3 (row 7)
        # j=2 -> k=2 (row 10), k=3 (row 11)
        # V_232 rows are j*2+k'.
        # j=0, k'=0 -> row 0. Maps to V_234 row 2 (j=0, k=2).
        # j=0, k'=1 -> row 1. Maps to V_234 row 3 (j=0, k=3).
        # j=1, k'=0 -> row 2. Maps to V_234 row 6 (j=1, k=2).
        # j=1, k'=1 -> row 3. Maps to V_234 row 7 (j=1, k=3).
        # j=2, k'=0 -> row 4. Maps to V_234 row 10 (j=2, k=2).
        # j=2, k'=1 -> row 5. Maps to V_234 row 11 (j=2, k=3).
        
        U_234[:, 10:20] = U_232
        
        # Map V
        # V_234[j*4 + (k'+2)] = V_232[j*2 + k']
        for j in range(3):
            for k_prime in range(2):
                idx_234 = j * 4 + (k_prime + 2)
                idx_232 = j * 2 + k_prime
                V_234[idx_234, 10:20] = V_232[idx_232, :]
                
        # W_234 rows: k*2+i.
        # Block 2 k is 2,3.
        # k=2, i=0 -> row 4. W_232 k'=0, i=0 -> row 0.
        # k=2, i=1 -> row 5. W_232 k'=0, i=1 -> row 1.
        # k=3, i=0 -> row 6. W_232 k'=1, i=0 -> row 2.
        # k=3, i=1 -> row 7. W_232 k'=1, i=1 -> row 3.
        
        # W_234[k*2 + i] for k=2,3 maps to W_234[4:8]
        # W_232[k'*2 + i] for k'=0,1 maps to W_232[0:4]
        # Note: The index mapping for W is direct if we just slice.
        # W_234[4:8, 10:20] = W_232[:, :]
        W_234[4:8, 10:20] = W_232
        
        # Verify final result
        # n=2, m=3, p=4
        n_final, m_final, p_final = 2, 3, 4
        U_final = U_234
        V_final = V_234
        W_final = W_234
        
        # Reshape for einsum
        U3 = U_final.reshape(n_final, m_final, 20)
        V3 = V_final.reshape(m_final, p_final, 20)
        W3 = W_final.reshape(p_final, n_final, 20)
        
        T_final = np.ones((n_final, m_final, p_final))
        A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        if np.allclose(A_approx, T_final, atol=1e-6):
            # Return lists
            return {
                "U": U_final.tolist(),
                "V": V_final.tolist(),
                "W": W_final.tolist()
            }
    else:
        # Fallback to schoolbook rank 24
        n, m, p = 2, 3, 4
        rank = 24
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
