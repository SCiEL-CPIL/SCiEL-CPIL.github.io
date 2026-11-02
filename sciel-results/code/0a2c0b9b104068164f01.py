import numpy as np
import random
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    time_limit = 110  # seconds
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Helper to check if values are close to half-integers
    def is_half_integer(matrix, tol=1e-9):
        for row in matrix:
            for val in row:
                nearest = round(val * 2) / 2
                if abs(val - nearest) > tol:
                    return False, None
        result = []
        for row in matrix:
            res_row = []
            for val in row:
                nearest = round(val * 2) / 2
                if nearest == int(nearest):
                    res_row.append(int(nearest))
                else:
                    res_row.append(nearest)
            result.append(res_row)
        return True, result

    def verify_decomposition(U, V, W, rank):
        # Check if sum_c U[i,j,c] V[j,k,c] W[i,k,c] == 1 for all i,j,k
        # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # We try to find a decomposition with rank R.
    # The condition is that for all (i, j, k), sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
    # This is a system of linear equations for W given U and V.
    # Let's iterate R from a lower bound up to schoolbook - 1.
    # Lower bound for 3x4x5 is likely > 20. Let's start from 25.
    
    # We will iterate ranks in increasing order to find the minimal one.
    # To save time, we don't iterate all ranks from 1 to 60.
    # We try specific ranks known to be promising or just a range.
    # Let's try ranks: 30, 35, 40, 45, 50.
    # If we find one, we return it.
    
    ranks_to_try = [30, 35, 40, 45, 50, 55]
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
            
        # Number of restarts per rank
        num_restarts = 200
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U and V with random values from {-1, 0, 1}
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(nm)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(mp)]
            
            # Build the linear system for W
            # Variables: W[k*n+i, c]
            # Equations: For each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Note: W index is k*n+i.
            # Let's group equations by the W variable index (k, i).
            # For a fixed (i, k), the variable is W_col = k*n + i.
            # The equations involving this variable are for all j.
            # Eq(j): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # This is a system of equations for the vector W.
            # Let's construct the matrix A and vector b.
            # Number of equations: n * m * p = 60.
            # Number of unknowns: R * n * p = R * 15.
            
            # A is sparse. We can use numpy lstsq.
            # A.shape = (60, R * 15)
            # b.shape = (60,)
            
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Variable index for W is k * n + i
                        # For each c, the coefficient is U[i*m+j, c] * V[j*p+k, c]
                        # The column in A is (k*n+i) * R + c ? No.
                        # Let's flatten W as a vector of length R * n * p.
                        # Index in W vector: w_idx = (k * n + i) * R + c
                        for c in range(R):
                            coeff = U_rand[i * m + j][c] * V_rand[j * p + k][c]
                            w_idx = (k * n + i) * R + c
                            A[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            # Solve A W_flat = b
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Reshape W_flat to (n*p, R) then transpose to (p*n, R) to match interface?
            # Interface W shape: (p*n, R). Index k*n+i.
            # Our w_idx was (k*n+i)*R + c.
            # So W_flat is flattened in row-major order of (p*n, R).
            W_sol = W_flat.reshape((p * n, R))
            
            # Check if W_sol is close to half-integers
            ok, W_nice = is_half_integer(W_sol)
            if ok:
                # Verify the decomposition with the rounded values
                # Convert U_rand, V_rand to lists if needed (they are already lists)
                # W_nice is a list of lists
                if verify_decomposition(U_rand, V_rand, W_nice, R):
                    best_rank = R
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_nice
                    break # Found a valid one for this rank, move to next lower rank? 
                    # Actually, we want the minimal rank. So we break this rank loop and try smaller?
                    # No, we are iterating ranks in increasing order. So if we find R=30, we return it.
                    return {"U": best_U, "V": best_V, "W": best_W}

        if time.time() - start_time > time_limit:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
