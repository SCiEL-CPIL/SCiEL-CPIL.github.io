"""Matrix multiplication 3x4x5 with rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return an exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    
    The output factor W follows the trace convention W[k*n+i, r].
    
    We first attempt to find a lower-rank decomposition via numerical search.
    If unsuccessful, we fall back to the schoolbook decomposition with rank n*m*p = 60.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook decomposition of rank n*m*p = 60
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def residuals(params):
        """
        Compute residuals for the tensor decomposition equations.
        params: flattened array of U, V, W parameters.
        """
        R = params.shape[0] // (n * m + m * p + p * n)
        if R == 0:
            return np.zeros(n * m * p)
        
        U = params[0 : n * m * R].reshape(n * m, R)
        V = params[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
        W = params[n * m * R + m * p * R :].reshape(p * n, R)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over rank r
                    res[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :]) * 1.0 # Wait, W is multiplied too?
                    # The equation is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
                    # This is a triple product.
                    # res[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is not a simple dot product of U and V.
                    pass
        
        # Correct calculation
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U[i * m + j, :]
                    v_vec = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    # sum_r u_vec[r] * v_vec[r] * w_val[r]
                    res[idx] = np.sum(u_vec * v_vec * w_val) - 1.0
                    idx += 1
        return res

    # Try to find lower rank
    best_rank = rank_school
    best_params = None
    
    # Search from rank_school - 1 down to a lower bound
    # Lower bound for 3x4x5 is not trivially known, but let's try down to 50
    for R in range(rank_school - 1, 49, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize parameters
        # A good initialization is to take the schoolbook solution and project it?
        # Or just random.
        # Let's try a structured initialization:
        # Use the first R terms of the schoolbook decomposition?
        # This might bias towards a sub-decomposition.
        
        U_init = np.zeros((n * m, R))
        V_init = np.zeros((m * p, R))
        W_init = np.zeros((p * n, R))
        
        # Copy first R terms from schoolbook
        for r in range(R):
            # Map schoolbook index r to (i, j, k)
            # Schoolbook order: j (0..3), i (0..2), k (0..4)
            # r = j * n * p + i * p + k
            k = r % p
            i = (r // p) % n
            j = r // (n * p)
            
            U_init[i * m + j, r] = 1.0
            V_init[j * p + k, r] = 1.0
            W_init[k * n + i, r] = 1.0
            
        # Add some random noise to escape local minima if we just copy
        # Or maybe try a few random restarts?
        # Given time, let's do one structured attempt per rank.
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            sol = least_squares(residuals, params_init, method='lm', max_nfev=1000)
            if sol.success and np.max(np.abs(sol.fun)) < 1e-8:
                # Check if we can round to half-integers
                params_opt = sol.x
                # Split back
                U_opt = params_opt[0 : n * m * R].reshape(n * m, R)
                V_opt = params_opt[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
                W_opt = params_opt[n * m * R + m * p * R :].reshape(p * n, R)
                
                # Round to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify exactness
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_round[i * m + j, :] * V_round[j * p + k, :] * W_round[k * n + i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-9:
                    best_rank = R
                    best_params = (U_round, V_round, W_round)
                    break # Found a valid lower rank, stop searching lower
        except Exception:
            continue
            
    if best_params is not None:
        U, V, W = best_params
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
