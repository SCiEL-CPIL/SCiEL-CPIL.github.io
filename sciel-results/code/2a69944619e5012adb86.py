"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
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
    
    # Try to reduce rank by identifying redundant columns.
    # A column r is redundant if for every (i,j,k) triple, the product
    # U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] can be replicated by other columns.
    # More practically: try removing one column at a time and check if the
    # resulting factorization still matches the target tensor.
    
    # The target tensor T[i,j,k] = 1 for all i,j,k (the bilinear map coefficient)
    # Actually, the correct interpretation: we need
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} ... no.
    # For matrix mult C[i,k] = sum_j A[i,j] B[j,k], the tensor entry is 1 for all (i,j,k).
    # So the constraint is: for all i,j,k: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    
    # Build the "tensor" as a 3D array
    T = np.ones((n, m, p))
    
    # Check current decomposition
    def check_decomposition(Um, Vm, Wm):
        """Check if sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(Um.shape[1]):
                        s += Um[i*m+j, r] * Vm[j*p+k, r] * Wm[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Greedy: try to remove columns one by one
    # For each column, check if removing it still gives exact decomposition
    # This is O(rank^2 * n*m*p) which is feasible for rank ~216
    
    current_rank = rank0
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    # Try removing columns in order
    removed = []
    for r_try in range(rank0):
        if time.time() > deadline - 10:
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
        
        # Quick check: compute tensor product
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(mask)):
                        s += U_test[i*m+j, r] * V_test[j*p+k, r] * W_test[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Remove this column
            U_cur = U_test
            V_cur = V_test
            W_cur = W_test
            current_rank = len(mask)
            removed.append(r_try)
    
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
