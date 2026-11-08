"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import numpy as np
from scipy.optimize import least_squares
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Precompute for vectorized equation evaluation
    eq_count = n * p * n * m * m * p
    rhs = np.zeros(eq_count)
    i_arr = np.zeros(eq_count, dtype=np.int32)
    k_arr = np.zeros(eq_count, dtype=np.int32)
    i2_arr = np.zeros(eq_count, dtype=np.int32)
    j_arr = np.zeros(eq_count, dtype=np.int32)
    j2_arr = np.zeros(eq_count, dtype=np.int32)
    k2_arr = np.zeros(eq_count, dtype=np.int32)
    
    idx = 0
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            i_arr[idx] = i
                            k_arr[idx] = k
                            i2_arr[idx] = i2
                            j_arr[idx] = j
                            j2_arr[idx] = j2
                            k2_arr[idx] = k2
                            if i2 == i and k2 == k and j == j2:
                                rhs[idx] = 1.0
                            idx += 1
    
    u_idx = (i2_arr * m + j_arr).astype(np.int32)
    v_idx = (j2_arr * p + k2_arr).astype(np.int32)
    w_idx = (k_arr * n + i_arr).astype(np.int32)
    
    def make_equations(target_rank):
        def equations(params):
            U_flat = params[0:target_rank*9].reshape(target_rank, 9)
            V_flat = params[target_rank*9:target_rank*21].reshape(target_rank, 12)
            W_flat = params[target_rank*21:target_rank*33].reshape(target_rank, 12)
            
            lhs = np.zeros(eq_count)
            for s in range(target_rank):
                lhs += W_flat[s, w_idx] * U_flat[s, u_idx] * V_flat[s, v_idx]
            return lhs - rhs
        return equations
    
    # Try to find low-rank decomposition, starting from lower ranks
    for target_rank in [24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline - 20:
            break
        
        eq_func = make_equations(target_rank)
        np.random.seed(42 + target_rank)
        
        best_result = None
        best_cost = float('inf')
        
        # Try a few random initializations
        for attempt in range(2):
            if time.time() > deadline - 25:
                break
            
            params0 = np.random.randn(target_rank * 33) * 0.5
            try:
                result = least_squares(eq_func, params0, method='lm', max_nfev=500)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result
                    if result.cost < 1e-8:
                        break
            except Exception:
                pass
        
        if best_result is not None and best_result.cost < 1e-6:
            params = best_result.x
            U_flat = params[0:target_rank*9].reshape(target_rank, 9)
            V_flat = params[target_rank*9:target_rank*21].reshape(target_rank, 12)
            W_flat = params[target_rank*21:target_rank*33].reshape(target_rank, 12)
            
            U_mat = U_flat.T
            V_mat = V_flat.T
            W_mat = W_flat.T
            
            # Try rounding to half-integers
            U_r = np.round(U_mat * 2) / 2
            V_r = np.round(V_mat * 2) / 2
            W_r = np.round(W_mat * 2) / 2
            
            # Verify
            lhs_check = np.zeros(eq_count)
            for s in range(target_rank):
                lhs_check += W_r[s, w_idx] * U_r[s, u_idx] * V_r[s, v_idx]
            max_err = np.max(np.abs(lhs_check - rhs))
            
            if max_err < 1e-6:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            else:
                return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    
    return schoolbook()
# EVOLVE-BLOCK-END
