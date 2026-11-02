import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook

    # Search for lower ranks
    # We try ranks from 59 down to 55. 
    # Ranks below 55 are less likely to be found by random search.
    # We use a limited number of attempts to avoid timeout.
    
    for r in range(59, 54, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        # We try different entry distributions
        distributions = [
            lambda: np.random.randint(-1, 2, (n * m, r)), # {-1, 0, 1}
            lambda: np.random.randint(0, 2, (n * m, r)),   # {0, 1}
            lambda: np.random.randint(-2, 3, (n * m, r)),  # {-2, -1, 0, 1, 2}
        ]
        
        found = False
        for dist_func in distributions:
            if found:
                break
            if time.time() > deadline:
                break
                
            for attempt in range(50):
                if time.time() > deadline:
                    break
                    
                U = dist_func()
                V = dist_func()
                
                W = np.zeros((p * n, r))
                valid = True
                
                for i in range(n):
                    for k in range(p):
                        # Build matrix A (m x r)
                        # A[j, c] = U[i*m + j, c] * V[j*p + k, c]
                        A = np.zeros((m, r))
                        for j in range(m):
                            u_row = U[i * m + j, :]
                            v_row = V[j * p + k, :]
                            A[j, :] = u_row * v_row
                        
                        # Solve A * w = 1
                        try:
                            # Use least squares
                            w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            
                            # Check if the solution is close to half-integers
                            # We check if 2*w is close to integers
                            w_scaled = w * 2
                            w_int = np.round(w_scaled)
                            
                            # Check error
                            if np.allclose(w_scaled, w_int, atol=1e-9):
                                w_half = w_int / 2.0
                                # Verify exactness
                                if np.allclose(A @ w_half, np.ones(m), atol=1e-9):
                                    W[k * n + i, :] = w_half
                                else:
                                    valid = False
                                    break
                            else:
                                # Try to find a nearby half-integer solution?
                                # For now, if it's not close, we fail.
                                valid = False
                                break
                        except:
                            valid = False
                            break
                            
                    if not valid:
                        break
                
                if valid:
                    # Verify the full decomposition
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for c in range(r):
                                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                                if abs(val - 1) > 1e-9:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        # We found a valid decomposition with rank r
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        
                        # Ensure half-integers are represented correctly
                        # Convert floats like 1.0 to 1, 0.5 to 0.5
                        # The problem accepts integer/half-integer.
                        # Numpy floats are fine.
                        
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better rank found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
