import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # We know schoolbook is 105.
    # Try to find lower rank.
    # Lower bound is roughly 36.
    # Let's try ranks from 40 to 104.
    
    for r in range(40, 105):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Random U, V with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, r))
            V = np.random.choice([-1, 0, 1], size=(m*p, r))
            
            # Solve for W
            # For each k, i, we have m equations:
            # sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
            # Let A_ki be the m x r matrix where A_ki[j, c] = U[i*m+j, c] * V[j*p+k, c]
            # We want A_ki * w_ki = 1 (vector of ones)
            
            W = np.zeros((p*n, r))
            valid = True
            for k in range(p):
                for i in range(n):
                    # Construct A_ki
                    A_ki = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_ki[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    
                    # Solve A_ki * w = 1
                    # Use least squares or pinv
                    try:
                        w_ki, residuals, rank, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if w_ki is close to half-integers
                    # Multiply by 2 to check for integers
                    w_ki_scaled = w_ki * 2
                    w_ki_rounded = np.round(w_ki_scaled)
                    
                    if np.allclose(w_ki_scaled, w_ki_rounded, atol=1e-6):
                        W[k*n+i, :] = w_ki_rounded / 2.0
                    else:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Verify the decomposition exactly
                # Check sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
                success = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(s - 1.0) > 1e-6:
                                success = False
                                break
                        if not success: break
                    if not success: break
                
                if success:
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
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
