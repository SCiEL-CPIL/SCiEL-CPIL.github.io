import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Target ranks to try
    # Known minimal rank for 2x3x4 is 19.
    # We try from 19 up to 23.
    target_ranks = [19, 20, 21, 22, 23]
    
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # Pre-compute indices for constraints
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # There are n*m*p = 24 constraints.
    
    def compute_residuals(U_flat, V_flat, W_flat, r):
        # U_flat is of size n*m*r
        # V_flat is of size m*p*r
        # W_flat is of size p*n*r
        residuals = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over c
                    u_part = U_flat[u_idx * r : (u_idx + 1) * r]
                    v_part = V_flat[v_idx * r : (v_idx + 1) * r]
                    w_part = W_flat[w_idx * r : (w_idx + 1) * r]
                    # Element-wise product sum
                    res = np.dot(u_part, v_part * w_part)
                    residuals[idx] = res - 1.0
                    idx += 1
        return residuals

    def try_rank(r, num_restarts=10):
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                return None
            
            # Random initialization
            # Scale: values around 1/sqrt(r) or similar?
            # Let's try small random values
            U_init = np.random.randn(n * m * r) * 0.5
            V_init = np.random.randn(m * p * r) * 0.5
            W_init = np.random.randn(p * n * r) * 0.5
            
            # Combine into one vector for least_squares
            x0 = np.concatenate([U_init, V_init, W_init])
            
            def fun(x):
                U_f = x[:n * m * r]
                V_f = x[n * m * r : n * m * r + m * p * r]
                W_f = x[n * m * r + m * p * r :]
                return compute_residuals(U_f, V_f, W_f, r)
            
            try:
                # Use scipy.optimize.least_squares
                import scipy.optimize as so
                result = so.least_squares(fun, x0, method='trf', max_nfev=1000)
                
                if result.success or result.cost < 1e-6:
                    U_f = result.x[:n * m * r]
                    V_f = result.x[n * m * r : n * m * r + m * p * r]
                    W_f = result.x[n * m * r + m * p * r :]
                    
                    # Reshape
                    U_sol = U_f.reshape(n * m, r)
                    V_sol = V_f.reshape(m * p, r)
                    W_sol = W_f.reshape(p * n, r)
                    
                    # Snap to half-integers
                    # Multiply by 2, round, divide by 2
                    U_snapped = np.round(U_sol * 2) / 2
                    V_snapped = np.round(V_sol * 2) / 2
                    W_snapped = np.round(W_sol * 2) / 2
                    
                    # Verify snapped solution
                    # Convert to list for consistency with return type, but check with numpy
                    U_check = U_snapped
                    V_check = V_snapped
                    W_check = W_snapped
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                val = np.dot(U_check[u_idx], V_check[v_idx] * W_check[w_idx])
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                    
                    if valid:
                        # Convert to lists of floats/ints
                        # The prompt accepts integer/half-integer entries.
                        # We return them as lists.
                        return {
                            "U": U_snapped.tolist(),
                            "V": V_snapped.tolist(),
                            "W": W_snapped.tolist()
                        }
            except Exception:
                pass
        return None

    for r in target_ranks:
        if time.time() - start_time > time_limit:
            break
        res = try_rank(r)
        if res is not None:
            best_U, best_V, best_W = res["U"], res["V"], res["W"]
            best_rank = r
            # If we found a lower rank, we can stop if it's the theoretical min (19)
            # or continue to see if we can go lower?
            # 19 is the known min for 2x3x4.
            if r == 19:
                break

    return {"U": best_U, "V": best_V, "W": best_W}
