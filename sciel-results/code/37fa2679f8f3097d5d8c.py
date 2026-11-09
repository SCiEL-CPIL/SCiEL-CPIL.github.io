"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Try to find a rank-minimal decomposition
    # Start with known constructions and search
    
    # For 2x4x4, try ranks from low to high
    # Known: schoolbook is 32. Try to find lower.
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, r
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = r_sb
    
    # Try to find a lower rank decomposition using random search + local optimization
    # The constraint: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    def check_decomp(U, V, W, tol=1e-9):
        """Check if decomposition is valid."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False, abs(s - 1.0)
        return True, 0.0
    
    def objective(U, V, W):
        """Sum of squared errors."""
        n, m, p = DIMENSIONS
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err += (s - 1.0) ** 2
        return err
    
    # Try structured search for lower ranks
    # Use the fact that for 2x4x4, we can potentially use rank 24 or lower
    
    # Attempt: use a parametrized family and optimize
    for target_rank in [24, 25, 26, 27, 28, 29, 30, 31]:
        if time.time() > deadline - 20:
            break
        
        found = False
        max_attempts = 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            r = target_rank
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize with random values
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Gradient descent on the objective
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradients numerically (expensive but simple)
                # Or use analytic gradient
                
                # Analytic gradient:
                # d/dU[i,j,c] = 2 * (s_ijc - 1) * V[j,k,c] * W[k,i,c]  (summed over k)
                # Actually: for each (i,j,k), the term is U[i,j,c]*V[j,k,c]*W[k,i,c]
                # d/dU[i,j,c] = sum_k 2*(s_ijc - 1)*V[j,k,c]*W[k,i,c]  -- no, s depends on c too
                
                # Let's do: err = sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
                # d err / d U[i,j,c] = sum_k 2*(sum_c' U[i,j,c']*V[j,k,c']*W[k,i,c'] - 1) * V[j,k,c] * W[k,i,c]
                
                # This is expensive. Let's use a simpler approach: alternating minimization
                
                # Fix V and W, solve for U (linear in each U entry? No, it's cubic)
                # Actually for fixed V,W: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k
                # This is linear in U! For fixed V,W, we have a linear system for U.
                
                # For fixed V,W: for each (i,j), we need sum_c U[i,j,c] * (V[j,k,c]*W[k,i,c]) = 1 for all k
                # So for each (i,j), we have p equations in r unknowns (U[i,j,0..r-1])
                # The coefficient matrix is: A[(k,c)] = V[j,k,c]*W[k,i,c], shape (p*r, r)
                # We want A * U[i,j,:] = 1 (vector of ones, length p)
                
                # Let's do alternating least squares
                for inner_step in range(10):
                    if time.time() > deadline - 5:
                        break
                    
                    # Update U: for each (i,j), solve least squares
                    for i in range(n):
                        for j in range(m):
                            # Build matrix: for each k, sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
                            # Coefficient for U[i,j,c] in equation k: V[j,k,c]*W[k,i,c]
                            # So A[k, c] = V[j,k,c]*W[k,i,c], shape (p, r)
                            A_mat = np.zeros((p, r))
                            for k in range(p):
                                for c in range(r):
                                    A_mat[k, c] = V[j * p + k, c] * W[k * n + i, c]
                            # Solve A_mat @ U[i,j,:] = ones(p)
                            try:
                                U[i * m + j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                            except:
                                pass
                    
                    # Update V: for each (j,k), solve least squares
                    for j in range(m):
                        for k in range(p):
                            # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i
                            # Coefficient for V[j,k,c] in equation i: U[i,j,c]*W[k,i,c]
                            # A[i, c] = U[i,j,c]*W[k,i,c], shape (n, r)
                            A_mat = np.zeros((n, r))
                            for i in range(n):
                                for c in range(r):
                                    A_mat[i, c] = U[i * m + j, c] * W[k * n + i, c]
                            try:
                                V[j * p + k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                            except:
                                pass
                    
                    # Update W: for each (k,i), solve least squares
                    for k in range(p):
                        for i in range(n):
                            # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all j
                            # Coefficient for W[k,i,c] in equation j: U[i,j,c]*V[j,k,c]
                            # A[j, c] = U[i,j,c]*V[j,k,c], shape (m, r)
                            A_mat = np.zeros((m, r))
                            for j in range(m):
                                for c in range(r):
                                    A_mat[j, c] = U[i * m + j, c] * V[j * p + k, c]
                            try:
                                W[k * n + i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                            except:
                                pass
                
                # Check convergence
                valid, err = check_decomp(U, V, W, tol=1e-8)
                if valid:
                    best_decomp = (U.copy(), V.copy(), W.copy())
                    best_rank = r
                    found = True
                    break
                
                if objective(U, V, W) < 1e-6:
                    valid, err = check_decomp(U, V, W, tol=1e-7)
                    if valid:
                        best_decomp = (U.copy(), V.copy(), W.copy())
                        best_rank = r
                        found = True
                        break
        
        if found:
            break
    
    U, V, W = best_decomp
    
    # Convert to lists with integer/half-integer if possible
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest half-integer if close
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-9:
                    new_row.append(rounded)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END

