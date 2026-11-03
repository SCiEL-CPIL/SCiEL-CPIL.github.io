import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Trivial decomposition for fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1

    # Build correct tensor for M_{3,3,3}
    # U idx: (rA, cA)
    # V idx: (cA, cB)
    # W idx: (cB, rA)
    T_333 = np.zeros((9, 9, 9))
    for rA in range(3):
        for cA in range(3):
            for cB in range(3):
                i_idx = rA * 3 + cA
                j_idx = cA * 3 + cB
                k_idx = cB * 3 + rA
                T_333[i_idx, j_idx, k_idx] = 1.0

    def find_decomp(T, r, max_iter=2000, num_restarts=10):
        """
        Attempts to find a rank-r decomposition of T with integer/half-integer entries.
        Returns (U, V, W) if found, else None.
        """
        best_error = np.inf
        best_U, best_V, best_W = None, None, None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + restart)
            # Initialize with small random values
            U_init = np.random.randn(9, r) * 0.1
            V_init = np.random.randn(9, r) * 0.1
            W_init = np.random.randn(9, r) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def loss(x):
                U = x[:9*r].reshape(9, r)
                V = x[9*r:18*r].reshape(9, r)
                W = x[18*r:].reshape(9, r)
                T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = T_calc - T
                return np.sum(err * err)
            
            try:
                res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
                if res.fun < 1e-4: # Threshold for "close enough" to try rounding
                    U = res.x[:9*r].reshape(9, r)
                    V = res.x[9*r:18*r].reshape(9, r)
                    W = res.x[18*r:].reshape(9, r)
                    
                    # Try rounding to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    T_calc_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    if np.allclose(T_calc_r, T, atol=1e-5):
                        return U_r, V_r, W_r
                    
                    # Try rounding to nearest integer
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    
                    T_calc_i = np.einsum('ir,jr,kr->ijk', U_i, V_i, W_i)
                    if np.allclose(T_calc_i, T, atol=1e-5):
                        return U_i, V_i, W_i
                        
            except Exception:
                pass
                
        return None

    best_U, best_V, best_W = None, None, None
    best_rank_333 = 27 # Fallback to trivial 27

    # Try ranks 21 to 26
    for r in range(21, 27):
        if time.time() > deadline - 10:
            break
        res = find_decomp(T_333, r, num_restarts=5)
        if res:
            best_U, best_V, best_W = res
            best_rank_333 = r
            break # Found a valid one, stop

    # Construct final decomposition for M_{3,3,7}
    # M_{3,3,1} trivial
    U_331 = np.zeros((9, 9))
    V_331 = np.zeros((9, 9))
    W_331 = np.zeros((3, 9)) # p=1, n=3 -> 3 rows
    col = 0
    for i in range(3):
        for j in range(3):
            U_331[i * 3 + j, col] = 1
            V_331[j * 1 + 0, col] = 1
            W_331[0 * 3 + i, col] = 1
            col += 1

    if best_U is not None:
        r333 = best_rank_333
        total_rank = 2 * r333 + 9
        
        U_final = np.zeros((9, total_rank))
        V_final = np.zeros((21, total_rank))
        W_final = np.zeros((21, total_rank))
        
        # Block 1: cols 0-2 of B (cB=0,1,2)
        U_final[:, 0:r333] = best_U
        V_final[0:9, 0:r333] = best_V
        W_final[0:9, 0:r333] = best_W
        
        # Block 2: cols 3-5 of B (cB=3,4,5)
        # V index in best_V is cA*3 + cB_local (0..2)
        # We need to map cB_global (3..5) to cB_local (0..2)
        # V_final row index: cA*3 + cB_global.
        # For cB_global in 3..5, cB_local = cB_global - 3.
        # So V_final[cA*3 + 3 + cB_local, ...] = best_V[cA*3 + cB_local, ...]
        # This is just a shift.
        V_final[9:18, r333:2*r333] = best_V
        W_final[9:18, r333:2*r333] = best_W
        U_final[:, r333:2*r333] = best_U
        
        # Block 3: col 6 of B (cB=6)
        U_final[:, 2*r333:2*r333+9] = U_331
        V_final[18:21, 2*r333:2*r333+9] = V_331
        W_final[18:21, 2*r333:2*r333+9] = W_331
        
        # Verify
        valid = True
        # Check a few random points or all points? 9*21*21 = 3969 points. Fast.
        for i in range(9):
            for j in range(21):
                for k in range(21):
                    val = 0
                    for r in range(total_rank):
                        val += U_final[i, r] * V_final[j, r] * W_final[k, r]
                    
                    # Determine expected value
                    # i = rA * 3 + cA
                    # j = cA * 3 + cB_global
                    # k = cB_global * 3 + rA
                    rA, cA = divmod(i, 3)
                    cA2, cB_global = divmod(j, 3)
                    cB2, rA2 = divmod(k, 3)
                    
                    if cA == cA2 and rA == rA2 and cB_global == cB2:
                        if not np.isclose(val, 1.0):
                            valid = False
                            break
                    else:
                        if not np.isclose(val, 0.0):
                            valid = False
                            break
                if not valid: break
            if not valid: break
            
        if valid:
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            
    # Fallback to trivial
    return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
