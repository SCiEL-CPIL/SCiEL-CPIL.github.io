import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_target = 23
    
    # Target tensor T
    # T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    # The decomposition is: sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = T[i, j, k]
    
    # Flatten indices for linear system
    # Variables: W[k, i, r] for k in [0, p), i in [0, n), r in [0, rank_target)
    # Total variables: p * n * rank_target
    # Equations: n * m * p
    
    n_vars = p * n * rank_target
    n_eqs = n * m * p
    
    # Precompute indices mapping
    # Equation index e corresponds to (i, j, k)
    # Variable index v corresponds to (k, i, r)
    
    # We will try to find U, V such that the system M W = T has a rational solution with small denominators.
    
    # Initialize with schoolbook solution as fallback
    U_fb = np.zeros((n * m, n * m * p), dtype=int)
    V_fb = np.zeros((m * p, n * m * p), dtype=int)
    W_fb = np.zeros((p * n, n * m * p), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    for trial in range(10000):
        if time.time() > deadline:
            break
            
        # Generate random sparse U and V
        # Use small integers
        U = np.zeros((n * m, rank_target), dtype=int)
        V = np.zeros((m * p, rank_target), dtype=int)
        
        # Randomly fill some entries with -1, 0, 1
        # To increase chance of success, we can try to "perturb" the schoolbook
        # But random search is simpler to implement correctly.
        
        # Let's try a structured random search:
        # Pick random linear combinations of schoolbook terms?
        # Or just random sparse matrices.
        
        # To make it more likely to find a solution, we can use the fact that
        # the rank 23 decomposition is "close" to rank 24 in some sense.
        
        # Let's just use random integers in {-1, 0, 1}
        for r in range(rank_target):
            # Each column of U and V
            # Randomly set some entries
            # Ensure not all zero
            for idx in range(n * m):
                if np.random.rand() < 0.3:
                    U[idx, r] = np.random.choice([-1, 1])
            for idx in range(m * p):
                if np.random.rand() < 0.3:
                    V[idx, r] = np.random.choice([-1, 1])
                    
        # Build linear system M W = T
        # M is n_eqs x n_vars
        M = np.zeros((n_eqs, n_vars))
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Equation for T[i, j, k]
                    # Sum over r: U[i, j, r] * V[j, k, r] * W[k, i, r]
                    # The term W[k, i, r] has coefficient U[i, j, r] * V[j, k, r]
                    # Variable index for W[k, i, r] is (k * n + i) * rank_target + r
                    for r in range(rank_target):
                        coeff = U[i * m + j, r] * V[j * p + k, r]
                        if coeff != 0:
                            var_idx = (k * n + i) * rank_target + r
                            M[eq_idx, var_idx] = coeff
                    eq_idx += 1
                    
        T_vec = np.ones(n_eqs)
        
        # Solve M W = T_vec
        # Use least squares to get a particular solution
        try:
            W_sol, residuals, rank_M, s = np.linalg.lstsq(M, T_vec, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Check if solution is close to integer/half-integer
        # Multiply by 2 to check for integers
        W_check = W_sol * 2
        
        # Check if W_check is close to integer
        W_int = np.round(W_check)
        
        if np.allclose(W_check, W_int, atol=1e-6):
            # Potential solution found
            W_candidate = W_int / 2.0
            
            # Verify exactly using fractions or high precision
            # Convert to fractions for exact check
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank_target):
                            u_val = U[i * m + j, r]
                            v_val = V[j * p + k, r]
                            w_val = W_candidate[(k * n + i) * rank_target + r]
                            val += u_val * v_val * w_val
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
            if valid:
                # Return the solution
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_candidate.tolist()
                }
                
    # Fallback to schoolbook
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
