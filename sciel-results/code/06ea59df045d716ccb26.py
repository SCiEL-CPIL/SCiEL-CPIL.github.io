import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135
    
    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Lower bound: max(n*m, m*p, p*n)
    lower_bound = max(n * m, m * p, p * n)
    
    # Try to find better decompositions using structured search
    # Strategy: Use Winograd-style embedding and random search with small integer/half-integer entries
    
    np.random.seed(42)
    
    # For each rank, try structured and random searches
    for R in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Number of attempts based on rank
        max_attempts = 30 if R < 60 else 15
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random U and V with small integer/half-integer entries
            # Use values from {-1, -0.5, 0, 0.5, 1}
            entries = np.array([-1, -0.5, 0, 0.5, 1])
            
            U_rand = np.random.choice(entries, size=(n * m, R))
            V_rand = np.random.choice(entries, size=(m * p, R))
            
            # For each (i, k), solve for W[k*n+i, :]
            # We need: for all j, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a system of m equations in R unknowns
            
            W_sol = np.zeros((p * n, R))
            valid = True
            
            for i in range(n):
                if not valid:
                    break
                for k in range(p):
                    if not valid:
                        break
                    
                    # Build m x R matrix where A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A_mat = np.zeros((m, R))
                    for j in range(m):
                        A_mat[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                    
                    # Solve A_mat * W_row = ones(m) using least squares
                    try:
                        W_row, residuals, rank, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        
                        # Try to round to nearest half-integer
                        W_row_rounded = np.round(W_row * 2) / 2
                        
                        # Check if rounded solution satisfies the equations
                        residual = np.max(np.abs(A_mat @ W_row_rounded - np.ones(m)))
                        
                        if residual < 1e-9:
                            W_sol[k * n + i, :] = W_row_rounded
                        else:
                            # Try the unrounded solution
                            residual_orig = np.max(np.abs(A_mat @ W_row - np.ones(m)))
                            if residual_orig < 1e-9:
                                # Check if we can find a half-integer solution nearby
                                W_sol[k * n + i, :] = W_row_rounded
                                residual2 = np.max(np.abs(A_mat @ W_sol[k * n + i, :] - np.ones(m)))
                                if residual2 > 1e-9:
                                    valid = False
                            else:
                                valid = False
                    except:
                        valid = False
            
            if not valid:
                continue
            
            # Verify the full solution
            all_ok = True
            for i in range(n):
                if not all_ok:
                    break
                for j in range(m):
                    if not all_ok:
                        break
                    for k in range(p):
                        val = np.dot(U_rand[i * m + j, :] * V_rand[j * p + k, :], W_sol[k * n + i, :])
                        if abs(val - 1) > 1e-9:
                            all_ok = False
                            break
            
            if all_ok:
                best_rank = R
                best_solution = {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_sol.tolist()
                }
                break
    
    return best_solution
