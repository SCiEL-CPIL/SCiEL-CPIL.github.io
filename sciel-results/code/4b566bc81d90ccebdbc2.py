import numpy as np
import time

def construct():
    n, m, p = 2, 5, 5
    # Target shapes:
    # U: (n*m, R) = (10, R)
    # V: (m*p, R) = (25, R)
    # W: (p*n, R) = (10, R)
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    deadline = time.time() + 140 # 10s safety margin
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook solution as fallback
    # R = 50
    R_school = n * m * p
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank
    # Based on sequence for 2xm x m: 1, 7, 14, 22... next might be 31, 32...
    # We will search from R=30 to R=49
    
    for R in range(30, 50):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random seeds
        for seed in range(20):
            if time.time() > deadline:
                break
                
            rng = np.random.RandomState(seed)
            
            # Initialize V and W with small integers/half-integers
            # Values in {-1, -0.5, 0, 0.5, 1}
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, R))
            W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, R))
            
            # Initialize U randomly as well, or solve for it
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, R))
            
            # Alternating Minimization with Exact Rounding
            for iter in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                # For each (i, j), we have system:
                # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
                # Let A[i,j,k,r] = V[j,k,r] * W[k,i,r]
                # Then sum_r U[i,j,r] * A[i,j,k,r] = 1
                # This is A[i,j] @ U[i,j] = 1_vector
                
                U_new = np.zeros((n * m, R))
                success_U = True
                for i in range(n):
                    for j in range(m):
                        # Construct matrix M of shape (p, R)
                        # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        # Solve M @ u = 1
                        try:
                            # Least squares solution
                            u_ls, residuals, rank, s = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            # Round to half-integers
                            u_int = np.round(u_ls * 2) / 2
                            
                            # Verify exactness
                            if np.allclose(M @ u_int, np.ones(p), atol=1e-9):
                                U_new[i * m + j, :] = u_int
                            else:
                                # Try small perturbations if direct rounding fails
                                # This is a heuristic repair
                                found_solution = False
                                for delta in [-0.5, 0, 0.5]:
                                    u_try = u_int + delta
                                    if np.allclose(M @ u_try, np.ones(p), atol=1e-9):
                                        U_new[i * m + j, :] = u_try
                                        found_solution = True
                                        break
                                if not found_solution:
                                    # Fall back to least squares rounded, mark as not exact
                                    # We keep the value but will check global error later
                                    U_new[i * m + j, :] = u_int
                                    success_U = False
                        except:
                            success_U = False
                            U_new[i * m + j, :] = U[i * m + j, :] # keep old
                
                U = U_new
                
                # Update V
                V_new = np.zeros((m * p, R))
                success_V = True
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, R))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        
                        try:
                            v_ls, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            v_int = np.round(v_ls * 2) / 2
                            
                            if np.allclose(M @ v_int, np.ones(n), atol=1e-9):
                                V_new[j * p + k, :] = v_int
                            else:
                                found_solution = False
                                for delta in [-0.5, 0, 0.5]:
                                    v_try = v_int + delta
                                    if np.allclose(M @ v_try, np.ones(n), atol=1e-9):
                                        V_new[j * p + k, :] = v_try
                                        found_solution = True
                                        break
                                if not found_solution:
                                    V_new[j * p + k, :] = v_int
                                    success_V = False
                        except:
                            success_V = False
                            V_new[j * p + k, :] = V[j * p + k, :]
                
                V = V_new
                
                # Update W
                W_new = np.zeros((p * n, R))
                success_W = True
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, R))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        try:
                            w_ls, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            w_int = np.round(w_ls * 2) / 2
                            
                            if np.allclose(M @ w_int, np.ones(m), atol=1e-9):
                                W_new[k * n + i, :] = w_int
                            else:
                                found_solution = False
                                for delta in [-0.5, 0, 0.5]:
                                    w_try = w_int + delta
                                    if np.allclose(M @ w_try, np.ones(m), atol=1e-9):
                                        W_new[k * n + i, :] = w_try
                                        found_solution = True
                                        break
                                if not found_solution:
                                    W_new[k * n + i, :] = w_int
                                    success_W = False
                        except:
                            success_W = False
                            W_new[k * n + i, :] = W[k * n + i, :]
                
                W = W_new
                
                # Check global convergence
                # Compute reconstruction
                recon = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                
                if np.allclose(recon, np.ones((n, m, p)), atol=1e-9):
                    # Found a valid decomposition
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
            
            if found:
                break
        
        if found:
            # Try to go lower?
            # We are iterating R from 30 up. If we found at R, we can break 
            # because we want the MINIMAL rank. 
            # Wait, the loop is `for R in range(30, 50)`.
            # If we find a solution at R=35, we should check if 34 or 33 works?
            # No, we are going UP. We should go DOWN or start from lower bound.
            # But we don't know the lower bound.
            # Strategy: Search from low to high. First found is minimal?
            # Yes, if we search R=30, 31, 32... the first one that works is the minimal rank in that range.
            # However, if 30 fails, 31 might work.
            # So we break immediately upon finding a solution.
            break

    # Convert to lists of lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
