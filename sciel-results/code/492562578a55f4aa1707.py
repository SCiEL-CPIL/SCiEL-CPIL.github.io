import time
import numpy as np
import random

DIMENSIONS = [2, 3, 7]
DEADLINE = 145.0
START_TIME = time.time()

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try ranks from lower bound to best_rank - 1
    # Theoretical lower bound for 2x3x7 is around 24-28
    min_rank_to_try = 24
    
    for R in range(min_rank_to_try, best_rank):
        if time.time() > DEADLINE - 8:
            break
            
        # Number of attempts scales with remaining time
        remaining_time = DEADLINE - 8 - time.time()
        max_attempts = max(50, int(remaining_time * 50))
        
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 5:
                break
                
            # Randomly initialize V and W with different strategies
            strategy = attempt % 4
            
            if strategy == 0:
                # Small integers
                V = np.random.randint(-2, 3, size=(m * p, R))
                W = np.random.randint(-2, 3, size=(p * n, R))
            elif strategy == 1:
                # Mostly zeros with some ones
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for idx in range(m * p * R // 3):
                    V[np.random.randint(0, m * p), np.random.randint(0, R)] = np.random.choice([-1, 1])
                for idx in range(p * n * R // 3):
                    W[np.random.randint(0, p * n), np.random.randint(0, R)] = np.random.choice([-1, 1])
            elif strategy == 2:
                # Structured: use Kronecker-like patterns
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for t in range(R):
                    # Assign random rows to each t
                    v_rows = np.random.choice(m * p, size=max(1, m * p // 3), replace=False)
                    w_rows = np.random.choice(p * n, size=max(1, p * n // 3), replace=False)
                    V[v_rows, t] = np.random.choice([-1, 1], size=len(v_rows))
                    W[w_rows, t] = np.random.choice([-1, 1], size=len(w_rows))
            else:
                # Random real values, then round later
                V = np.random.randn(m * p, R)
                W = np.random.randn(p * n, R)
                # Scale to keep values manageable
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
            
            U = np.zeros((n * m, R))
            
            valid = True
            # Solve for U for each j
            for j in range(m):
                # Build the linear system M * u = b
                # M has shape (n * p, n * R)
                # Row index: (i, k) -> i * p + k
                # Col index: (i, t) -> i * R + t
                # Coefficient M[i*p+k, i*R+t] = V[j*p+k, t] * W[k*n+i, t]
                
                M = np.zeros((n * p, n * R))
                b = np.ones(n * p)
                
                # Vectorized construction
                for i in range(n):
                    for k in range(p):
                        row_idx = i * p + k
                        # M[row_idx, i*R:(i+1)*R] = V[j*p+k, :] * W[k*n+i, :]
                        M[row_idx, i * R:(i + 1) * R] = V[j * p + k, :] * W[k * n + i, :]
                
                # Solve M * u = b
                try:
                    u, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Reshape u back to (n, R)
                u_matrix = u.reshape((n, R))
                
                # Try rounding to integers
                u_int = np.round(u_matrix).astype(int)
                
                # Check if u_int satisfies the equations exactly
                # Vectorized check
                residual_int = 0
                for i in range(n):
                    for k in range(p):
                        val = np.dot(u_int[i, :], V[j * p + k, :] * W[k * n + i, :])
                        if abs(val - 1) > 1e-9:
                            residual_int += 1
                            break
                    if residual_int > 0:
                        break
                
                if residual_int > 0:
                    # Try half-integers
                    u_half = np.round(u_matrix * 2) / 2.0
                    residual_half = 0
                    for i in range(n):
                        for k in range(p):
                            val = np.dot(u_half[i, :], V[j * p + k, :] * W[k * n + i, :])
                            if abs(val - 1) > 1e-9:
                                residual_half += 1
                                break
                        if residual_half > 0:
                            break
                    if residual_half > 0:
                        valid = False
                        break
                    else:
                        u_matrix = u_half
                else:
                    u_matrix = u_int
                
                # Store U for this j
                for i in range(n):
                    U[i * m + j, :] = u_matrix[i, :]
            
            if valid:
                # Final verification of the whole tensor
                full_valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            if abs(val - 1) > 1e-9:
                                full_valid = False
                                break
                        if not full_valid:
                            break
                    if not full_valid:
                        break
                
                if full_valid:
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

    return best_result
