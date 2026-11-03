import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Schoolbook
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Try to find lower rank
    for r in range(rank_school - 1, max(rank_school - 5, 0), -1):
        if time.time() > deadline:
            break
            
        # Initialize with schoolbook minus one column
        U = U_sb[:, :r].copy()
        V = V_sb[:, :r].copy()
        W = W_sb[:, :r].copy()
        
        # Flatten
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residuals(x):
            U_flat = x[:n*m*r]
            V_flat = x[n*m*r : n*m*r + m*p*r]
            W_flat = x[n*m*r + m*p*r :]
            
            U_mat = U_flat.reshape(n*m, r)
            V_mat = V_flat.reshape(m*p, r)
            W_mat = W_flat.reshape(p*n, r)
            
            res = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Sum over t
                        # U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                        res[i,j,k] = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
            
            return (res - 1).flatten()
        
        try:
            # Use least_squares
            sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
            
            if sol.success or sol.cost < 1e-6:
                # Check if we can round to integers/half-integers
                x_sol = sol.x
                U_sol = x_sol[:n*m*r].reshape(n*m, r)
                V_sol = x_sol[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_sol = x_sol[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Round to nearest 0.5
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                # Verify exactness
                res_check = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res_check[i,j,k] = np.dot(U_round[i*m+j, :], V_round[j*p+k, :] * W_round[k*n+i, :])
                            
                if np.allclose(res_check, 1.0, atol=1e-9):
                    best_rank = r
                    best_U = U_round
                    best_V = V_round
                    best_W = W_round
                    # If we found a lower rank, we can break or continue to find even lower
                    # Given the difficulty, breaking is safer to save time
                    break 
        except Exception:
            pass
            
    # Convert to lists of lists of numbers
    def to_list(mat):
        return [[float(val) for val in row] for row in mat]
        
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
