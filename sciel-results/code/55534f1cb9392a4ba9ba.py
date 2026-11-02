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

    # Attempt to find a lower rank using randomized search with integer snapping
    # We search for ranks slightly below 48
    for R in range(47, 44, -1):
        if time.time() > deadline:
            break
            
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization with small integers
            U_r = np.random.randint(-1, 2, size=(n*m, R))
            V_r = np.random.randint(-1, 2, size=(m*p, R))
            
            valid = True
            W_r = np.zeros((p*n, R))
            
            # Solve for W rows
            for i in range(n):
                for k in range(p):
                    # We need sum_c U[i*m+j, c] V[j*p+kp, c] W[k*n+i, c] = delta(k, kp)
                    # For fixed i, k, we solve for W_row = W[k*n+i, :]
                    # The system is: A @ W_row = b
                    # A has rows for each (j, kp)
                    # b has entries 1 if k==kp else 0
                    
                    A_mat = np.zeros((m*p, R))
                    b_vec = np.zeros(m*p)
                    
                    idx = 0
                    for j in range(m):
                        for kp in range(p):
                            for c in range(R):
                                A_mat[idx, c] = U_r[i*m+j, c] * V_r[j*p+kp, c]
                            b_vec[idx] = 1.0 if k == kp else 0.0
                            idx += 1
                    
                    # Solve least squares
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if solution is close to integer or half-integer
                    # The problem allows integer/half-integer
                    # Check distance to nearest 0.5
                    snapped = np.round(sol * 2) / 2
                    if not np.allclose(sol, snapped, atol=1e-6):
                        valid = False
                        break
                    
                    W_r[k*n+i, :] = snapped
            
            if not valid:
                continue
            
            # Verify exactness
            full_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for kp in range(p):
                            val = 0.0
                            for c in range(R):
                                val += U_r[i*m+j, c] * V_r[j*p+kp, c] * W_r[k*n+i, c]
                            expected = 1.0 if k == kp else 0.0
                            if not np.isclose(val, expected, atol=1e-9):
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
