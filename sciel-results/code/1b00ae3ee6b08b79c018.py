import numpy as np
import random
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook
    U = np.zeros((n*m, best_rank))
    V = np.zeros((m*p, best_rank))
    W = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U, V, W

    # Try to find lower rank
    for R in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            U_r = np.random.randint(-2, 3, size=(n*m, R))
            V_r = np.random.randint(-2, 3, size=(m*p, R))
            
            # Check if we can solve for W
            valid = True
            W_r = np.zeros((p*n, R))
            
            for i in range(n):
                for k in range(p):
                    # Build system for W[k*n+i, :]
                    # Equations: for each j, k'
                    # sum_c U[i*m+j, c] * V[j*p+k', c] * W[k*n+i, c] = delta(k, k')
                    
                    A_mat = np.zeros((m*p, R))
                    b_vec = np.zeros(m*p)
                    
                    idx = 0
                    for j in range(m):
                        for kp in range(p):
                            # Row idx of A_mat
                            for c in range(R):
                                A_mat[idx, c] = U_r[i*m+j, c] * V_r[j*p+kp, c]
                            b_vec[idx] = 1 if k == kp else 0
                            idx += 1
                    
                    # Solve A_mat @ W_row = b_vec
                    # Use least squares
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if solution is integer
                    if not np.allclose(sol, np.round(sol), atol=1e-9):
                        valid = False
                        break
                    
                    W_r[k*n+i, :] = np.round(sol)
            
            if not valid:
                continue
            
            # Verify
            full_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for kp in range(p):
                            val = 0
                            for c in range(R):
                                val += U_r[i*m+j, c] * V_r[j*p+kp, c] * W_r[k*n+i, c]
                            expected = 1 if k == kp else 0
                            if val != expected:
                                full_valid = False
                                break
                        if not full_valid: break
                    if not full_valid: break
                if not full_valid: break
            
            if full_valid:
                best_rank = R
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
