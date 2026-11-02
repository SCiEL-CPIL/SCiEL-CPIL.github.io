import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 32 # Schoolbook
    best_solution = None
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100 # 100s limit to be safe
    
    # Try ranks from 27 up to 31
    for R in range(27, 32):
        if time.time() > deadline:
            break
            
        # Multiple restarts
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W randomly
            # Using small random values to help convergence
            U = np.random.randn(n, R)
            V = np.random.randn(m, R)
            W = np.random.randn(p, R)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Solve for U: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # Reshape T to (n, m*p)
                T_reshaped = T.reshape(n, m * p)
                # Construct A_U such that A_U @ U_flat = T_flat
                # A_U[i, (j,k)*R + r] = V[j,r] * W[k,r]
                # This is a block diagonal or similar structure.
                # More efficiently:
                # For each i, T[i, :, :] = sum_r U[i,r] (V[:,r] * W[:,r])^T
                # Let M_r = V[:,r] * W[:,r] (size m x p)
                # T[i] = sum_r U[i,r] M_r
                # This is a linear system: [M_1 ... M_R] @ U[:,r] = T[i]
                # The matrix is of size (m*p) x R
                A_U = np.zeros((m * p, R))
                for r in range(R):
                    M_r = V[:, r] * W[:, r]
                    A_U[:, r] = M_r.flatten()
                
                # Solve A_U @ U_new = T_reshaped
                U_new, residuals, rank, sv = np.linalg.lstsq(A_U, T_reshaped, rcond=None)
                U = U_new
                
                # Similarly for V
                T_reshaped_V = T.reshape(m, n * p) # Note: T is [i,j,k], V is [j,r]
                # T[i,j,k] = sum_r V[j,r] U[i,r] W[k,r]
                # For fixed j, T[:, j, :] = sum_r V[j,r] (U[:,r] * W[:,r])^T
                A_V = np.zeros((n * p, R))
                for r in range(R):
                    M_r = U[:, r] * W[:, r]
                    A_V[:, r] = M_r.flatten()
                
                V_new, residuals, rank, sv = np.linalg.lstsq(A_V, T_reshaped_V, rcond=None)
                V = V_new
                
                # Similarly for W
                T_reshaped_W = T.reshape(p, n * m)
                # T[i,j,k] = sum_r W[k,r] U[i,r] V[j,r]
                # For fixed k, T[:,:,k] = sum_r W[k,r] (U[:,r] * V[:,r])^T
                A_W = np.zeros((n * m, R))
                for r in range(R):
                    M_r = U[:, r] * V[:, r]
                    A_W[:, r] = M_r.flatten()
                
                W_new, residuals, rank, sv = np.linalg.lstsq(A_W, T_reshaped_W, rcond=None)
                W = W_new
                
            # Check if solution is close to half-integers
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify exactness
            # Compute sum_r U_rounded[i,r] V_rounded[j,r] W_rounded[k,r]
            # This is the tensor product sum
            # Efficiently:
            # C[i,k] = sum_j (sum_r U[i,r] V[j,r] W[k,r])? No.
            # The condition is: for all i,j,k, sum_r U[i,r] V[j,r] W[k,r] == 1
            # Let's check a few random indices or all indices (32 is small)
            is_exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U_rounded[i, r] * V_rounded[j, r] * W_rounded[k, r]
                        if abs(val - 1.0) > 1e-9:
                            is_exact = False
                            break
                    if not is_exact:
                        break
                if not is_exact:
                    break
            
            if is_exact:
                # Found a valid decomposition
                # Convert to list of lists
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback to schoolbook
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
