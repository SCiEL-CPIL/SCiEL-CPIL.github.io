import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, R) = (8, R)
    # V: (m*p, R) = (16, R)
    # W: (p*n, R) = (8, R)
    
    # Target Tensor T[i, j, k] = 1 if i, j, k correspond to A[i,j] * B[j,k] -> C[k,i]
    # Wait, the convention is W[k*n+i, r].
    # C[k, i] = sum_j A[i, j] B[j, k].
    # So T[i, j, k] = 1.
    # U index: i*m + j. (0..7)
    # V index: j*p + k. (0..15)
    # W index: k*n + i. (0..7)
    
    T_flat = np.zeros(n*m*p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                # The tensor is sum_r U[u_idx, r] V[v_idx, r] W[w_idx, r]
                # We want this to equal 1 for all i,j,k.
                # So T_flat is not directly the tensor, we need to form the matricization.
                
    # We can define the objective as the sum of squares of (Computed - Target)
    # Computed[k*n+i] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This is not linear in r.
    
    # Let's define a function that takes U, V, W and computes the error.
    
    def compute_error(params, R):
        U = params[0:R*n*m].reshape(R, n*m).T # (n*m, R)
        V = params[R*n*m:R*(n*m+m*p)].reshape(R, m*p).T # (m*p, R)
        W = params[R*(n*m+m*p):].reshape(R, p*n).T # (p*n, R)
        
        # Compute C = A B
        # C[k, i] = sum_j A[i, j] B[j, k]
        # We don't have A and B, we have the decomposition.
        # The decomposition says:
        # For each r, we have a rank-1 term.
        # The sum over r of U[:, r] outer V[:, r] outer W[:, r] should equal the tensor.
        # Let's check the entries.
        # Entry (i, j, k) is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # We want this to be 1 for all i, j, k.
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    errors.append(val - 1.0)
        return np.array(errors)

    # This is slow. Let's vectorize.
    # We can form the tensor T_computed = sum_r U_r outer V_r outer W_r.
    # Then check if T_computed == 1.
    
    def compute_error_vec(params, R):
        U = params[0:R*n*m].reshape(n*m, R)
        V = params[R*n*m:R*(n*m+m*p)].reshape(m*p, R)
        W = params[R*(n*m+m*p):].reshape(p*n, R)
        
        # T_computed[i*m+j, j*p+k, k*n+i] ? No, dimensions don't match directly.
        # We need to check all triples (i, j, k).
        # There are n*m*p = 32 triples.
        # We can precompute the indices.
        
        # U_idx[i, j] = i*m + j
        # V_idx[j, k] = j*p + k
        # W_idx[k, i] = k*n + i
        
        # val[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # We can compute this by iterating over r?
        # No, iterate over i, j, k.
        
        errors = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = i*m + j
                    v_row = j*p + k
                    w_row = k*n + i
                    errors[idx] = np.dot(U[u_row, :] * V[v_row, :], W[w_row, :]) - 1.0
                    idx += 1
        return errors

    # Optimization
    best_result = None
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try ranks from 16 to 20
    for R in range(16, 21):
        if time.time() > deadline:
            break
            
        # Try a few restarts
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Initialize
            # Use random normal
            np.random.seed(42 + R * 100 + restart)
            U0 = np.random.randn(n*m, R)
            V0 = np.random.randn(m*p, R)
            W0 = np.random.randn(p*n, R)
            
            # Gauge fixing: Scale U to have norm 1?
            # Let's try to fix the first element of each U column to 1?
            # If U0[u, r] is close to 0, this is bad.
            # Let's just use random init and hope.
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = least_squares(compute_error_vec, params0, args=(R,), method='lm', max_nfev=10000)
            except Exception:
                continue
                
            if res.cost < 1e-8:
                # Check if we can round to half-integers
                U_opt = res.x[0:R*n*m].reshape(n*m, R)
                V_opt = res.x[R*n*m:R*(n*m+m*p)].reshape(m*p, R)
                W_opt = res.x[R*(n*m+m*p):].reshape(p*n, R)
                
                # Round to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify exactness
                # Re-compute error with rounded values
                U_r = U_round
                V_r = V_round
                W_r = W_round
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_r[i*m+j, :] * V_r[j*p+k, :], W_r[k*n+i, :])
                            if abs(val - 1.0) > 1e-9:
                                max_err = 1.0
                                break
                        if max_err > 0: break
                    if max_err > 0: break
                    
                if max_err == 0:
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                
    # If no low-rank solution found, return the Strassen rank-28 solution
    # (Fallback)
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_block(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                gi = r
                gj = a_col_off * 2 + c
                U[u_idx(gi, gj), col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                gj = b_row_off * 2 + r
                gk = b_col_off * 2 + c
                V[v_idx(gj, gk), col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                gi = r
                gk = c_col_off * 2 + c
                W[w_idx(gk, gi), col] += coeff
                
    apply_block(0, 0, 0, 0, 0)
    apply_block(1, 1, 0, 0, 7)
    apply_block(0, 0, 1, 1, 14)
    apply_block(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
