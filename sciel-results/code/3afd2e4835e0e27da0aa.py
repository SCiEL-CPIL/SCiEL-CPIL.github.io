"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j][col] = 1
                V_school[j*p + k][col] = 1
                W_school[k*n + i][col] = 1
                col += 1
    
    # Try to find lower rank by solving linear systems
    # For fixed U and V, W must satisfy:
    # For each (i,k): sum_j sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # This is linear in W for fixed U,V
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try ranks from 35 to 59
    for r in range(35, 60):
        if time.time() > deadline:
            break
        
        # Try several random U,V initializations
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Random sparse U, V
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            
            # Initialize with random +/-1 with some sparsity
            for idx in range(n*m):
                for c in range(r):
                    if np.random.random() < 0.4:
                        U[idx, c] = np.random.choice([-1, 1])
            for idx in range(m*p):
                for c in range(r):
                    if np.random.random() < 0.4:
                        V[idx, c] = np.random.choice([-1, 1])
            
            # For each (i,k), we have a linear equation in W[k*n+i, :]
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
            # Wait, this isn't quite right. The equation is:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each (i,j,k)
            # For fixed (i,k), this gives m equations in the r variables W[k*n+i, 0..r-1]
            # But W[k*n+i, c] is shared across j, so we have a system:
            # For fixed (i,k): for each j, sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
            # This is a system of m equations in r unknowns (W[k*n+i, :])
            
            valid = True
            W = np.zeros((p*n, r))
            
            for i in range(n):
                for k in range(p):
                    # Build the m x r system
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    
                    # Solve A @ W_row = b
                    try:
                        W_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if exact solution
                        if np.allclose(A @ W_row, b, atol=1e-10):
                            W[k*n + i, :] = W_row
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Verify and convert to list
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c] for c in range(r))
                            if abs(s - 1) > 1e-10:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    best_rank = r
                    best_U = U_list
                    best_V = V_list
                    best_W = W_list
                    break  # Found a solution for this rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
