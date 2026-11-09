"""Improved program with structured search for lower rank decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # 1. Baseline: Schoolbook Decomposition (Rank 60)
    rank_baseline = n * m * p
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = rank_baseline
    deadline = start_time + time_limit - 3.0
    
    # Try to find lower rank using structured approaches
    # Strategy: Use random matrices with small integer entries, then solve for W
    
    for R in range(rank_baseline - 1, 39, -1):
        if time.time() > deadline:
            break
            
        # Number of trials per rank - increase for higher ranks
        trials = 2000 if R > 50 else 1000
        
        for t in range(trials):
            if time.time() > deadline:
                break
                
            # Generate random U and V with entries from {-2, -1, 0, 1, 2}
            # This gives more flexibility than just {-1, 0, 1}
            U = np.random.randint(-2, 3, size=(n * m, R))
            V = np.random.randint(-2, 3, size=(m * p, R))
            
            # For each (i, k) pair, solve for W row
            W_candidate = np.zeros((p * n, R), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Construct A of shape (m, R)
                    # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, R), dtype=float)
                    for j in range(m):
                        A[j, :] = U[i * m + j, :].astype(float) * V[j * p + k, :].astype(float)
                    
                    # Solve A * w = ones(m)
                    # First check if system is consistent
                    try:
                        # Use least squares to get a solution
                        w_ls, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        
                        # Check if this solution is close to half-integers
                        w_half = np.round(w_ls * 2) / 2.0
                        
                        # Verify the solution
                        if np.allclose(A.dot(w_half), np.ones(m), atol=1e-8):
                            W_candidate[k * n + i, :] = w_half
                        else:
                            # Try to find if there's ANY solution in half-integer lattice
                            # Check consistency first
                            A_aug = np.hstack((A, np.ones((m, 1))))
                            if np.linalg.matrix_rank(A) != np.linalg.matrix_rank(A_aug):
                                valid = False
                                break
                            else:
                                # System is consistent but lstsq didn't give nice solution
                                # Try to find a solution in the null space
                                # Get null space basis
                                if rank_A < R:
                                    # Find null space
                                    V_svd = sv.T
                                    null_basis = V_svd[rank_A:]
                                    
                                    # Try a few random combinations in null space
                                    found = False
                                    for _ in range(20):
                                        y = np.random.randn(null_basis.shape[0])
                                        w_try = w_ls + null_basis.T.dot(y)
                                        w_try_half = np.round(w_try * 2) / 2.0
                                        if np.allclose(A.dot(w_try_half), np.ones(m), atol=1e-8):
                                            W_candidate[k * n + i, :] = w_try_half
                                            found = True
                                            break
                                    if not found:
                                        valid = False
                                        break
                                else:
                                    valid = False
                                    break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                            
                if not valid:
                    break
            
            if valid:
                # Verify the full tensor equation
                valid_full = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(R):
                                s += U[i * m + j, c] * V[j * p + k, c] * W_candidate[k * n + i, c]
                            if abs(s - 1) > 1e-7:
                                valid_full = False
                                break
                        if not valid_full:
                            break
                    if not valid_full:
                        break
                
                if valid_full:
                    best_solution = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_candidate.tolist()
                    }
                    best_rank = R
                    break  # Found solution for this R, move to R-1

    return best_solution
