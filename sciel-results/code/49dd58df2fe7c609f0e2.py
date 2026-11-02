import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
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

    # Try to find a better rank
    # We know rank must be at least max(n*m, m*p, p*n) = max(12, 20, 15) = 20.
    # But realistically, it's close to 60.
    # Let's try ranks from 55 down to 60.
    for r in range(55, 61):
        if time.time() > deadline:
            break
            
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Generate random U and V with entries 0 or 1
            # To ensure solvability, we might need structure.
            # Random 0/1 matrices often lead to full rank systems.
            U = np.random.randint(0, 2, (n * m, r))
            V = np.random.randint(0, 2, (m * p, r))
            
            W_valid = True
            W = np.zeros((p * n, r))
            
            # For each (i, k), solve for W[k, i, :]
            for i in range(n):
                for k in range(p):
                    # Build matrix M (m x r)
                    # M[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    M = np.zeros((m, r))
                    for j in range(m):
                        u_row = U[i * m + j, :]
                        v_row = V[j * p + k, :]
                        M[j, :] = u_row * v_row
                        
                    # Solve M * w = 1
                    # w is r-vector
                    # We want integer/half-integer solutions.
                    # First, check if system is consistent.
                    # Use least squares to get a candidate, then check if it's close to half-integers.
                    try:
                        # Add small noise to avoid singular matrices? No, M is fixed.
                        # If M is rank deficient, there might be multiple solutions.
                        # We want a solution with entries in {0, 0.5, 1}.
                        # Let's try to find a solution in the null space of the augmented matrix?
                        # Simpler: use np.linalg.lstsq
                        w, residuals, rank, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        
                        # Check if w is close to half-integers
                        w_rounded = np.round(w * 2) / 2
                        if np.allclose(M @ w_rounded, np.ones(m), atol=1e-9):
                            # Check if entries are valid
                            if np.all((w_rounded == 0) | (w_rounded == 0.5) | (w_rounded == 1)):
                                W[k * n + i, :] = w_rounded
                            else:
                                W_valid = False
                                break
                        else:
                            W_valid = False
                            break
                    except:
                        W_valid = False
                        break
                
                if not W_valid:
                    break
            
            if W_valid:
                # Verify the full decomposition
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            if abs(val - 1) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to lists for return
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    # Ensure integer/half-integer types if possible, though float is usually accepted if exact
                    # The prompt says "integer/half-integer entries", so we should return numbers that look like that.
                    # Numpy float 0.5 is fine.
                    return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
