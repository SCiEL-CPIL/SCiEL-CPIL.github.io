import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We search for a solution with integer entries in the scaled system (factor 2)
    # Original entries are half-integers.
    # Equation: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    # Scaled: sum_c U'[i,j,c] V'[j,k,c] W'[k,i,c] = 8
    # U', V', W' are integers.
    
    # Lower bound estimate: max(n,m,p) + min(n,m,p) - 1 = 7 + 2 - 1 = 8.
    # Try r from 8 upwards.
    
    start_time = time.time()
    max_time = 110.0  # Safety margin
    
    for r in range(8, 30):
        if time.time() - start_time > max_time:
            break
            
        # Number of restarts depends on r
        # Smaller r is harder, try more restarts? Or smaller r is less likely to exist?
        # Let's try 500 restarts per r.
        num_restarts = 500
        
        for restart in range(num_restarts):
            if time.time() - start_time > max_time:
                break
                
            # Initialize with small random integers
            # U: (n*m, r) = (8, r)
            # V: (m*p, r) = (28, r)
            # W: (p*n, r) = (14, r)
            U = np.random.randint(-3, 4, size=(n*m, r))
            V = np.random.randint(-3, 4, size=(m*p, r))
            W = np.random.randint(-3, 4, size=(p*n, r))
            
            # Iterative refinement
            converged = False
            for it in range(50):
                # Update U
                # For each i, j:
                # M[k, c] = V[j*p + k, c] * W[k*n + i, c]
                # Solve M @ U_col = 8
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r))
                        for k in range(p):
                            M[k, :] = V[j*p + k, :] * W[k*n + i, :]
                        # Solve M @ x = 8
                        # Using least squares
                        try:
                            x, residuals, rank, s = np.linalg.lstsq(M, np.ones(p) * 8, rcond=None)
                            U[i*m + j, :] = np.round(x).astype(int)
                        except:
                            pass
                
                # Update V
                # For each j, k:
                # N[i, c] = U[i*m + j, c] * W[k*n + i, c]
                # Solve N @ V_col = 8
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, r))
                        for i in range(n):
                            N[i, :] = U[i*m + j, :] * W[k*n + i, :]
                        try:
                            x, residuals, rank, s = np.linalg.lstsq(N, np.ones(n) * 8, rcond=None)
                            V[j*p + k, :] = np.round(x).astype(int)
                        except:
                            pass
                
                # Update W
                # For each k, i:
                # P[j, c] = U[i*m + j, c] * V[j*p + k, c]
                # Solve P @ W_col = 8
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, r))
                        for j in range(m):
                            P[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        try:
                            x, residuals, rank, s = np.linalg.lstsq(P, np.ones(m) * 8, rcond=None)
                            W[k*n + i, :] = np.round(x).astype(int)
                        except:
                            pass
                
                # Check convergence
                # Compute T = sum_c U V W
                # T[i, j, k] should be 8
                # Efficient check:
                # For each i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # We can vectorize this.
                # Create tensor T of shape (n, m, p)
                # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                
                # Loop is fine for small dims
                is_valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                            if val != 8:
                                is_valid = False
                                break
                        if not is_valid:
                            break
                    if not is_valid:
                        break
                
                if is_valid:
                    converged = True
                    break
                    
            if converged:
                # Convert back to original scale (divide by 2)
                U_orig = (U / 2.0).tolist()
                V_orig = (V / 2.0).tolist()
                W_orig = (W / 2.0).tolist()
                
                # Verify one last time with original values
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U_orig[i*m + j][c] * V_orig[j*p + k][c] * W_orig[k*n + i][c]
                            if not np.isclose(val, 1.0, atol=1e-9):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U_orig, "V": V_orig, "W": W_orig}

    # If no solution found, return schoolbook (rank 56)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
