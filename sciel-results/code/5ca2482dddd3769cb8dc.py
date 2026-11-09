import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as guaranteed fallback
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
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    T_target = np.ones((n, m, p))
    
    # Try ranks from just below schoolbook down to a reasonable minimum
    # For 2x5x6, try down to rank 20
    for r in range(rank_school - 1, 19, -1):
        if time.time() > deadline - 15:
            break
        
        # Multiple random restarts for each rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            
            seed = r * 1000 + restart * 7 + 42
            np.random.seed(seed)
            
            # Initialize with small random values
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Alternating minimization
            max_iters = 300
            prev_error = float('inf')
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), solve U[i*m+j,:] * M_U = T_target[i,j,:]
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(M.T, T_target[i, j, :], rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve V[j*p+k,:] * M_V = T_target[:,j,k]
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, r))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(M.T, T_target[:, j, k], rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), solve W[k*n+i,:] * M_W = T_target[i,:,k]
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, r))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(M.T, T_target[i, :, k], rcond=None)[0]
                        except:
                            pass
                
                # Check convergence every 10 iterations
                if it % 10 == 9:
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                                error += (val - 1.0) ** 2
                    if error < 1e-12:
                        # Verify it's a valid decomposition
                        best_rank = r
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        found = True
                        break
                    if abs(error - prev_error) < 1e-14:
                        # Stagnated
                        break
                    prev_error = error
            
            if found:
                break
        
        if found:
            break
    
    # Convert to lists
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
