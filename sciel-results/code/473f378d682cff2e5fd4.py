"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U0 = np.zeros((n * m, rank_school))
    V0 = np.zeros((m * p, rank_school))
    W0 = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin before 150s limit
    
    # Use Strassen-style block decomposition for 3x5x5
    # Split into blocks: A is 3x5, B is 5x5
    # We can use a recursive/block approach
    # For 5x5x5, known ranks are around 20-25 (much better than 125)
    # For 3x5x5, we can potentially do better
    
    # Let's try a direct approach with alternating least squares
    # The constraint is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U0, V0, W0
    
    # Try different target ranks
    # For 3x5x5, the minimum rank is at least max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # But it's likely higher. Let's try ranks from 74 down to 25
    
    target_ranks = list(range(rank_school - 1, 24, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            np.random.seed(attempt * 1000 + target_rank)
            U_curr = np.random.randn(n * m, target_rank) * 0.5
            V_curr = np.random.randn(m * p, target_rank) * 0.5
            W_curr = np.random.randn(p * n, target_rank) * 0.5
            
            # Alternating least squares
            converged = False
            max_iterations = 20
            
            for iteration in range(max_iterations):
                if time.time() > deadline:
                    break
                
                # Update U: fix V, W
                # For each (i,j): sum_k sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                # This is a linear system in U[i*m+j, :]
                U_new = np.zeros_like(U_curr)
                for i in range(n):
                    for j in range(m):
                        # Build matrix A where A[c, c'] = sum_k V[j*p+k,c] * W[k*n+i,c']
                        A_mat = np.zeros((target_rank, target_rank))
                        for k in range(p):
                            A_mat += np.outer(V_curr[j*p+k, :], W_curr[k*n+i, :])
                        # Solve A_mat @ U_new[i*m+j, :] = 1
                        try:
                            U_new[i*m+j, :] = np.linalg.solve(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank))
                        except:
                            U_new[i*m+j, :] = np.linalg.lstsq(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
                U_curr = U_new
                
                # Update V: fix U, W
                # For each (j,k): sum_i sum_c (U[i*m+j,c] * W[k*n+i,c]) * V[j*p+k,c] = 1
                V_new = np.zeros_like(V_curr)
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((target_rank, target_rank))
                        for i in range(n):
                            A_mat += np.outer(U_curr[i*m+j, :], W_curr[k*n+i, :])
                        try:
                            V_new[j*p+k, :] = np.linalg.solve(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank))
                        except:
                            V_new[j*p+k, :] = np.linalg.lstsq(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
                V_curr = V_new
                
                # Update W: fix U, V
                # For each (k,i): sum_j sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
                W_new = np.zeros_like(W_curr)
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((target_rank, target_rank))
                        for j in range(m):
                            A_mat += np.outer(U_curr[i*m+j, :], V_curr[j*p+k, :])
                        try:
                            W_new[k*n+i, :] = np.linalg.solve(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank))
                        except:
                            W_new[k*n+i, :] = np.linalg.lstsq(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
                W_curr = W_new
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_curr[i*m+j, :] * V_curr[j*p+k, :] * W_curr[k*n+i, :])
                            err += (val - 1)**2
                
                if err < 1e-10:
                    converged = True
                    break
            
            if converged:
                best_rank = target_rank
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                break
    
    # If we found a lower rank, use it; otherwise use schoolbook
    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U0.tolist()
        V_out = V0.tolist()
        W_out = W0.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
