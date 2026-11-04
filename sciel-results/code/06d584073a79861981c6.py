"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 * 3x7.
    Uses structured search with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin from 120s limit
    
    # Schoolbook solution: rank = n * m * p = 3 * 3 * 7 = 63
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Verify schoolbook solution
    def verify(U, V, W, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank solutions
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Try ranks from 21 upward, but be strategic about which ranks to try
    
    # Strategy: Try a few specific rank values with structured U, V
    # For each rank R, try random integer U, V in {-1, 0, 1} and solve for W
    
    search_deadline = start_time + 100  # Spend at most 100s on search
    
    # Try ranks that are likely to work based on structure
    # The 7 columns of B suggest we might need at least 7 * something
    # Try: 21, 22, 24, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 62
    
    candidate_ranks = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 62]
    
    for R in candidate_ranks:
        if time.time() > search_deadline:
            break
        if R >= best_rank:
            break
        
        found = False
        # Multiple trials per rank with different random seeds
        for trial in range(20):
            if time.time() > search_deadline - 5:
                break
            
            np.random.seed(trial * 7919 + R * 104729)
            
            # Generate random U, V with entries in {-1, 0, 1}
            U_r = np.random.randint(-1, 2, size=(n * m, R))
            V_r = np.random.randint(-1, 2, size=(m * p, R))
            
            # Solve for W: for each (k, i), we need
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # This gives a linear system in W[k*n+i, :]
            
            W_r = np.zeros((p * n, R), dtype=float)
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # Build system: for each j, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # A @ w = ones(m)
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U_r[i * m + j, :] * V_r[j * p + k, :]
                    
                    # Check if system is consistent
                    # Use least squares
                    try:
                        w_sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if the solution actually satisfies the equations
                    if np.max(np.abs(A @ w_sol - np.ones(m))) > 1e-8:
                        valid = False
                        break
                    
                    W_r[k * n + i, :] = w_sol
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W is close to integer or half-integer
            W_rounded = np.round(W_r * 2) / 2
            if not np.allclose(W_r, W_rounded, atol=1e-6):
                continue
            
            # Verify the complete solution
            if verify(U_r, V_r, W_rounded, tol=1e-6):
                best_U, best_V, best_W = U_r, V_r, W_rounded
                best_rank = R
                found = True
                break
        
        if found:
            continue  # Try smaller R
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
