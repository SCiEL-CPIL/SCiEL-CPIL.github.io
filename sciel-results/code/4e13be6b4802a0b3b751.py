import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target: Rank 15
    target_rank = 15
    
    # Time limit guard
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # Initialize U, V
    # We want integer/half-integer solutions.
    # Let's try random integers {-1, 0, 1}
    
    best_solution = None
    best_rank = n * m * p # 18
    
    # Fallback schoolbook
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Try to find rank 15
    for _ in range(200): # Max iterations
        if time.time() - start_time > time_limit:
            break
            
        # Random init
        U = np.random.randint(-1, 2, (n*m, target_rank))
        V = np.random.randint(-1, 2, (m*p, target_rank))
        
        # Solve for W
        # Eq: sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1
        # Flatten W to vector w of size p*n*target_rank
        # This is a linear system A w = b
        
        # Construct A
        # Rows: (i, j, k) -> 18 rows
        # Cols: (k, i, t) -> 6*15 = 90 cols
        A_mat = np.zeros((n*m*p, p*n*target_rank))
        b_vec = np.ones(n*m*p)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(target_rank):
                        # W index: k*n + i, col t
                        w_idx = (k*n + i) * target_rank + t
                        A_mat[idx, w_idx] = U[i*m+j, t] * V[j*p+k, t]
                    idx += 1
        
        # Solve A w = b
        # Use least squares
        try:
            w_sol, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W_flat = w_sol.reshape((p*n, target_rank))
            
            # Round to nearest 0.5
            W_rounded = np.round(W_flat * 2) / 2
            
            # Verify
            # Compute sum_t U V W_rounded
            # Reshape U, V, W for tensor product check
            # U: (n*m, R) -> (n, m, R)
            # V: (m*p, R) -> (m, p, R)
            # W: (p*n, R) -> (p, n, R)
            
            U_t = U.reshape(n, m, target_rank)
            V_t = V.reshape(m, p, target_rank)
            W_t = W_rounded.reshape(p, n, target_rank)
            
            # Sum over t: U[i,j,t] V[j,k,t] W[k,i,t]
            # This is a contraction.
            # result[i,j,k] = sum_t U_t[i,j,t] V_t[j,k,t] W_t[k,i,t]
            
            # Use einsum
            # 'ijt,jkt,kit->ijk'
            result = np.einsum('ijt,jkt,kit->ijk', U_t, V_t, W_t)
            
            # Check if result is all 1s
            if np.allclose(result, 1.0, atol=1e-9):
                # Found a solution!
                # Convert to list of lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_rounded.tolist()
                
                # Ensure integers are int, halves are float
                # The prompt allows integer/half-integer.
                return {"U": U_list, "V": V_list, "W": W_list}
                
        except Exception as e:
            continue
            
    # If not found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
