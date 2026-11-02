import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 100.0  # 50s margin before 150s kill

    def get_errors(U, V, W):
        """
        Returns a vector of errors for each term in the tensor identity.
        Identity:
        sum_r U[i'*m+j', r] V[j''*p+k', r] W[k*n+i, r] = delta_{i', i} delta_{k', k} delta_{j', j''}
        
        We compute the error for every valid (i, k, i', k', j', j'') tuple.
        There are n*p*n*p*m*m = 1024 such tuples.
        """
        U_arr = np.asarray(U)
        V_arr = np.asarray(V)
        W_arr = np.asarray(W)
        R = U_arr.shape[1]
        
        errors = np.zeros(1024)
        idx = 0
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for i_prime in range(n):
                    for k_prime in range(p):
                        for j_prime in range(m):
                            u_idx = i_prime * m + j_prime
                            for j_double_prime in range(m):
                                v_idx = j_double_prime * p + k_prime
                                target = 1.0 if (i == i_prime and k == k_prime and j_prime == j_double_prime) else 0.0
                                val = np.dot(U_arr[u_idx] * V_arr[v_idx], W_arr[w_idx])
                                errors[idx] = val - target
                                idx += 1
        return errors

    def check_identity(U, V, W, R):
        """
        Check if the factors satisfy the tensor identity.
        Returns max absolute error.
        """
        errors = get_errors(U, V, W)
        return np.max(np.abs(errors))

    def strassen_rank_28():
        """
        Constructs the known rank-28 Strassen-based solution.
        Returns U, V, W as numpy arrays.
        """
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        components = [
            # M1
            ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
            # M2
            ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
            # M3
            ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
            # M4
            ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
            # M5
            ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
            # M6
            ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
            # M7
            ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
        ]
        
        terms = [
            (0, 0, 0, 0), # A0, B00, C0
            (2, 2, 0, 0), # A1, B10, C0
            (0, 0, 2, 2), # A0, B01, C1
            (2, 2, 2, 2), # A1, B11, C1
        ]
        
        col = 0
        for (a_co, b_ro, b_co, c_co) in terms:
            for (u_c, v_c, w_c) in components:
                for (li, lj, val) in u_c:
                    gi = li
                    gj = a_co + lj
                    U[gi * m + gj, col] += val
                for (li, lj, val) in v_c:
                    gi = b_ro + li
                    gj = b_co + lj
                    V[gi * p + gj, col] += val
                for (li, lj, val) in w_c:
                    gi = li
                    gj = c_co + lj
                    W[(c_co + lj) * n + li, col] += val
                col += 1
        
        return U, V, W

    def try_rank_27_search():
        """
        Attempts to find a rank-27 decomposition.
        Strategy:
        1. Start from rank-28 solution.
        2. Zero out one column (or a linear combination) to create a rank-27 candidate.
        3. Optimize the remaining parameters to minimize the sum of squared errors.
        4. Check if the max error is sufficiently small.
        """
        R = 27
        U28, V28, W28 = strassen_rank_28()
        
        from scipy.optimize import minimize
        
        # We will try zeroing out each of the 28 columns in the rank-28 solution
        # and optimizing the remaining 27.
        # This is a heuristic: if the rank-28 solution is "close" to a rank-27 solution,
        # removing one "redundant" column and adjusting the others might close the gap.
        
        best_err = float('inf')
        best_params = None
        
        # Pre-compute flattened rank-28 solution for reference
        U28_flat = U28.flatten()
        V28_flat = V28.flatten()
        W28_flat = W28.flatten()
        
        for r_to_remove in range(28):
            if time.time() > deadline - 5:
                break
                
            # Create initial guess for rank 27 by removing column r_to_remove
            U_init = np.delete(U28, r_to_remove, axis=1)
            V_init = np.delete(V28, r_to_remove, axis=1)
            W_init = np.delete(W28, r_to_remove, axis=1)
            
            # Flatten for optimizer
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U_flat = params[:R*nm].reshape(R, nm)
                V_flat = params[R*nm:R*nm+R*mp].reshape(R, mp)
                W_flat = params[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(R, pn)
                
                # Use sum of squares for smoothness
                U_t = U_flat.T
                V_t = V_flat.T
                W_t = W_flat.T
                errors = get_errors(U_t, V_t, W_t)
                return np.sum(errors**2)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-14})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
                    # Check max error for this candidate
                    U_opt = best_params[:R*nm].reshape(R, nm).T
                    V_opt = best_params[R*nm:R*nm+R*mp].reshape(R, mp).T
                    W_opt = best_params[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(R, pn).T
                    max_err = check_identity(U_opt, V_opt, W_opt, R)
                    if max_err < 1e-6:
                        # Found a solution!
                        # Round to half-integers
                        U_r = np.round(U_opt * 2) / 2.0
                        V_r = np.round(V_opt * 2) / 2.0
                        W_r = np.round(W_opt * 2) / 2.0
                        final_err = check_identity(U_r, V_r, W_r, R)
                        if final_err < 1e-9:
                            return U_r, V_r, W_r
            except Exception:
                pass
        
        # If we found a good float solution but not rounded yet, try rounding the best one
        if best_params is not None and best_err < 1e-4:
            U_opt = best_params[:R*nm].reshape(R, nm).T
            V_opt = best_params[R*nm:R*nm+R*mp].reshape(R, mp).T
            W_opt = best_params[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(R, pn).T
            
            U_r = np.round(U_opt * 2) / 2.0
            V_r = np.round(V_opt * 2) / 2.0
            W_r = np.round(W_opt * 2) / 2.0
            
            err = check_identity(U_r, V_r, W_r, R)
            if err < 1e-9:
                return U_r, V_r, W_r
                
        return None

    # 1. Try to find rank 27
    result_27 = try_rank_27_search()
    if result_27 is not None:
        U, V, W = result_27
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # 2. Fallback to rank 28 (Strassen)
    U, V, W = strassen_rank_28()
    
    # Verify Strassen solution just in case
    err = check_identity(U, V, W, 28)
    if err > 1e-9:
        # This shouldn't happen, but fallback to schoolbook
        R = n * m * p
        U_s = np.zeros((nm, R))
        V_s = np.zeros((mp, R))
        W_s = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1
                    V_s[j*p + k, col] = 1
                    W_s[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
