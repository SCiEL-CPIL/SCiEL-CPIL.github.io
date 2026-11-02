import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook initialization
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

    # Try to reduce rank
    # We try ranks from R_school-1 down to a lower bound
    lower_bound = 29
    for R in range(R_school - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
            
        # Try a few random subsets of columns to keep
        # Number of attempts depends on how many columns we are dropping
        drop_count = R_school - R
        attempts = 10 if drop_count > 2 else 5
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            # Randomly select R columns to keep
            keep_indices = np.random.choice(R_school, R, replace=False)
            
            # Initialize with kept columns
            U_init = np.zeros((n * m, R))
            V_init = np.zeros((m * p, R))
            W_init = np.zeros((p * n, R))
            
            for new_idx, old_idx in enumerate(keep_indices):
                U_init[:, new_idx] = U_school[:, old_idx]
                V_init[:, new_idx] = V_school[:, old_idx]
                W_init[:, new_idx] = W_school[:, old_idx]
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residual(x):
                U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R + m*p*R], x[n*m*R + m*p*R:]
                U_mat = U_flat.reshape(n * m, R)
                V_mat = V_flat.reshape(m * p, R)
                W_mat = W_flat.reshape(p * n, R)
                
                res = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U_mat[i * m + j]
                            v_val = V_mat[j * p + k]
                            w_val = W_mat[k * n + i]
                            res[idx] = np.dot(u_val, v_val * w_val) - 1.0
                            idx += 1
                return res

            try:
                # Use trf method for large problems
                result = least_squares(residual, x0, method='trf', max_nfev=2000)
                
                # Check if solution is good
                if np.linalg.norm(result.fun) < 1e-5:
                    x_sol = result.x
                    U_sol = x_sol[:n*m*R].reshape(n * m, R)
                    V_sol = x_sol[n*m*R:n*m*R + m*p*R].reshape(m * p, R)
                    W_sol = x_sol[n*m*R + m*p*R:].reshape(p * n, R)
                    
                    # Round to nearest 0.5
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    if verify(U_int, V_int, W_int, n, m, p, R):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
            except Exception:
                pass

    # Fallback
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def verify(U, V, W, n, m, p, R):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
