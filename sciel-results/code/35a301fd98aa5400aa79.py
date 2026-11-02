import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    deadline = time.time() + 145  # Safety margin before 150s kill

    np.random.seed(42)
    
    # Precompute index mappings for efficiency
    # eq_idx = i*m*p + j*p + k
    # var_idx = (k*n + i)*R + r
    
    for R in range(20, best_rank):
        if time.time() > deadline:
            break
            
        # Number of random trials per rank - increase for smaller ranks
        trials = 5000 if R <= 25 else 2000
        
        found = False
        for trial in range(trials):
            if time.time() > deadline:
                break
                
            # Random U, V with small integer entries
            # Use entries in {-2, -1, 0, 1, 2} for more flexibility
            U_try = np.random.randint(-2, 3, size=(n * m, R))
            V_try = np.random.randint(-2, 3, size=(m * p, R))
            
            # Construct the linear system for W
            # A_mat[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
            # where eq_idx = i*m*p + j*p + k
            # and var_idx = (k*n + i)*R + r
            
            A_mat = np.zeros((n * m * p, p * n * R))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        var_base = (k * n + i) * R
                        # Vectorized: A_mat[eq_idx, var_base:var_base+R] = U_try[i*m+j, :] * V_try[j*p+k, :]
                        A_mat[eq_idx, var_base:var_base+R] = U_try[i * m + j, :] * V_try[j * p + k, :]
            
            B_vec = np.ones(n * m * p)
            
            # Solve A_mat * W_flat = B_vec
            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, B_vec, rcond=None)
            except:
                continue
                
            # Check residual
            if residuals is None or len(residuals) == 0:
                continue
            
            if residuals[0] > 1e-9:
                continue
                
            # Check if W_flat can be rounded to half-integers
            W_rounded = np.round(W_flat * 2) / 2.0
            
            # Verify exactness with rounded W using vectorized operations
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        var_base = (k * n + i) * R
                        sum_val = np.dot(U_try[i * m + j, :], V_try[j * p + k, :] * W_rounded[var_base:var_base+R])
                        if abs(sum_val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = R
                best_U = U_try.tolist()
                best_V = V_try.tolist()
                W_mat = W_rounded.reshape((p * n, R))
                best_W = W_mat.tolist()
                found = True
                break
        
        if found and R == 20:
            break  # Found optimal
        if found:
            break  # Found a solution, stop searching higher ranks
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
