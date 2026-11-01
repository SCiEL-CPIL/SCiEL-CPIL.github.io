"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to find rank 32
    target_rank = 32
    if time.time() < deadline - 30:
        # Strategy: Fix V and W to be sparse random matrices, solve for U.
        # We need V and W such that the system for U is solvable.
        # The system is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # For a fixed r, let M_r be the matrix of coefficients for U.
        # This is a linear system A * u = b, where A is (n*m*p) x (n*m*R)? No.
        # Unknowns are U[i*m+j, r] for all i,j,r. Total n*m*R unknowns.
        # Equations are for each (i,j,k). Total n*m*p equations.
        # Matrix A has shape (n*m*p) x (n*m*R).
        # A[(i,j,k), (i',j',r)] = V[j*p+k, r] * W[k*n+i, r] if i==i' and j==j' else 0.
        # This is a sparse matrix.
        
        # We can try to find V, W such that A has a solution.
        # Random V, W might not work.
        # We try a few random sparse V, W.
        
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Generate random sparse V and W
            V_rand = np.zeros((m * p, target_rank))
            W_rand = np.zeros((p * n, target_rank))
            
            # Initialize with some non-zeros
            for r in range(target_rank):
                # Pick random indices for V and W
                idx_v = random.randint(0, m * p - 1)
                idx_w = random.randint(0, p * n - 1)
                V_rand[idx_v, r] = 1
                W_rand[idx_w, r] = 1
            
            # Build linear system for U
            # Unknowns: U_flat of size n*m*R
            # Equations: n*m*p
            A = np.zeros((n * m * p, n * m * target_rank))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Coefficient for U[i*m+j, r] is V[j*p+k, r] * W[k*n+i, r]
                        for r in range(target_rank):
                            coeff = V_rand[j * p + k, r] * W_rand[k * n + i, r]
                            if coeff != 0:
                                # Unknown index: (i*m+j) * target_rank + r
                                A[eq_idx, (i * m + j) * target_rank + r] = coeff
                        eq_idx += 1
            
            # Solve A * u = b
            try:
                # Use least squares or pinv if underdetermined
                u, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                if residuals and residuals[0] < 1e-9:
                    # Check if solution is valid (integer/half-integer)
                    U_sol = u.reshape((n * m, target_rank))
                    # Check if entries are close to 0.5 multiples
                    if np.allclose(U_sol * 2, np.round(U_sol * 2), atol=1e-6):
                        # Verify full decomposition
                        # T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s_val = np.dot(U_sol[i*m+j], V_rand[j*p+k] * W_rand[k*n+i])
                                    if abs(s_val - 1.0) > 1e-9:
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        if valid:
                            best_U, best_V, best_W = U_sol, V_rand, W_rand
                            best_rank = target_rank
                            break # Found a valid one
            except Exception:
                pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
