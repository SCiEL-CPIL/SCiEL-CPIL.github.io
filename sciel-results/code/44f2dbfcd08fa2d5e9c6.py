import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
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
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = delta_{i,k} for matrix multiplication
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 49 down to 40 (known lower bounds suggest ~49-50 for 3x4x5)
    for rank in range(49, 39, -1):
        if time.time() > deadline - 5:
            break
        
        # Multiple random restarts
        for attempt in range(50):
            if time.time() > deadline - 3:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Use least squares to solve for the tensor matching
            # We want: sum_r U[:,r] outer V[:,r] outer W[:,r] to match target tensor
            # This is a bilinear problem, so we use alternating least squares
            
            converged = False
            for iteration in range(200):
                if time.time() > deadline - 2:
                    break
                
                # Update U: for each (i,j), find U[i*m+j, :] that minimizes
                # ||sum_r U[i*m+j,r] * (V[j*p+..., :] * W[..., i]) - target||^2
                for i in range(n):
                    for j in range(m):
                        # Target: for each k, T[i,j,k] = 1 if i==k else 0
                        # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta_{i,k}
                        # This is: U[i*m+j, :] @ diag(V[j*p+k, r] * W[k*n+i, r]) = delta
                        # Stack for all k:
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.zeros(p)
                        for k in range(p):
                            b[k] = 1.0 if k == i else 0.0
                        
                        # Solve A @ U_row = b
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), find V[j*p+k, :] that minimizes
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.zeros(n)
                        for i in range(n):
                            b[i] = 1.0 if i == k else 0.0
                        
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), find W[k*n+i, :] that minimizes
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.zeros(m)
                        b[:] = 1.0  # All entries should be 1 for the diagonal condition
                        # Wait, this isn't right. Let me reconsider.
                        
                        # Actually T[i,j,k] = 1 if i==k, 0 otherwise
                        # So for fixed (k,i), we need sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta_{i,k}
                        # This means: W[k*n+i, :] @ (U[i*m+0, :] * V[0*p+k, :] + ... + U[i*m+(m-1), :] * V[(m-1)*p+k, :]) = delta_{i,k}
                        
                        A2 = np.zeros(m)
                        for j in range(m):
                            A2[j] = 1.0  # We're solving for W, so the coefficient is the product of U and V
                        # Actually: sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}
                        # Let C[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                        # Then C @ W[k*n+i, :] = delta_{i,k}
                        
                        C = np.zeros(rank)
                        for j in range(m):
                            C += U[i * m + j, :] * V[j * p + k, :]
                        
                        # We need C @ W_row = delta_{i,k}
                        # This is one equation: C dot W_row = delta_{i,k}
                        # With rank variables and 1 equation, we have underdetermined system
                        # Use lstsq to find minimum norm solution
                        if abs(np.linalg.norm(C)) > 1e-10:
                            # W_row should satisfy C @ W_row = delta
                            # Minimum norm: W_row = delta * C / (C @ C)
                            W[k * n + i, :] = (1.0 if i == k else 0.0) * C / (np.dot(C, C) + 1e-12)
                        else:
                            W[k * n + i, :] = 0.0
                
                # Check convergence
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            target = 1.0 if i == k else 0.0
                            error += (val - target) ** 2
                
                if error < 1e-10:
                    converged = True
                    break
            
            if converged:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break  # Found a decomposition at this rank, try lower
    
    # Use the best found
    U_final, V_final, W_final = best_U, best_V, best_W
    rank_final = best_rank
    
    # Convert to lists for the required format
    U_list = [[float(U_final[i][j]) for j in range(rank_final)] for i in range(n * m)]
    V_list = [[float(V_final[i][j]) for j in range(rank_final)] for i in range(m * p)]
    W_list = [[float(W_final[i][j]) for j in range(rank_final)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
