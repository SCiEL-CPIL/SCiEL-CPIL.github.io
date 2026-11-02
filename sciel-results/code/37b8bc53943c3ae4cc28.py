"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Schoolbook: rank = n*m*p
    rank0 = n * m * p
    U = np.zeros((n * m, rank0))
    V = np.zeros((m * p, rank0))
    W = np.zeros((p * n, rank0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    def check_decomposition(Um, Vm, Wm):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
        R = Um.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute dot product efficiently
                    s = np.dot(Um[i*m+j], Vm[j*p+k] * Wm[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Greedy: try to remove columns one by one
    current_rank = rank0
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    # Try removing columns in order
    removed = set()
    for r_try in range(rank0):
        if time.time() > deadline - 15:
            break
        if r_try in removed:
            continue
        
        # Temporarily remove column r_try
        mask = [r for r in range(current_rank) if r != r_try]
        if not mask:
            break
        
        U_test = U_cur[:, mask]
        V_test = V_cur[:, mask]
        W_test = W_cur[:, mask]
        
        if check_decomposition(U_test, V_test, W_test):
            # Remove this column
            U_cur = U_test
            V_cur = V_test
            W_cur = W_test
            current_rank = len(mask)
            removed.add(r_try)
    
    # Final verification
    if not check_decomposition(U_cur, V_cur, W_cur):
        # Fallback to schoolbook
        U_final = U
        V_final = V
        W_final = W
    else:
        U_final = U_cur
        V_final = V_cur
        W_final = W_cur
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
# EVOLVE-BLOCK-END
