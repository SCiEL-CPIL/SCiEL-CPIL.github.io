import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Try ranks from low to high
    # Known bounds suggest < 90 is possible.
    # Try 50, 55, 60, 65, 70, 75, 80, 85, 89
    for R in range(50, 90, 5):
        if time.time() > deadline - 10:
            break
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            U, V, W = als_decompose(n, m, p, R, seed=42+attempt)
            if U is not None:
                return {"U": U, "V": V, "W": W}
            
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

def als_decompose(n, m, p, R, max_iter=200, seed=0):
    np.random.seed(seed)
    
    # Initialize
    U = np.random.rand(n * m, R) * 0.1
    V = np.random.rand(m * p, R) * 0.1
    W = np.random.rand(p * n, R) * 0.1
    
    best_err = float('inf')
    
    for it in range(max_iter):
        # Solve for W
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    W_col, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k * n + i, :] = W_col
                except:
                    pass
                    
        # Solve for U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    for r in range(R):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    U_col, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i * m + j, :] = U_col
                except:
                    pass
                    
        # Solve for V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    for r in range(R):
                        A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    V_col, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j * p + k, :] = V_col
                except:
                    pass
                    
        # Check error
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err = abs(val - 1)
                    if err > max_err:
                        max_err = err
                        
        if max_err < best_err:
            best_err = max_err
            if max_err < 1e-6:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r]
                            if abs(val - 1) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    # Clean up floats
                    for i in range(len(U_list)):
                        for j in range(len(U_list[i])):
                            if abs(U_list[i][j] - round(U_list[i][j])) < 1e-9:
                                U_list[i][j] = int(round(U_list[i][j]))
                            else:
                                U_list[i][j] = round(U_list[i][j] * 2) / 2
                    for i in range(len(V_list)):
                        for j in range(len(V_list[i])):
                            if abs(V_list[i][j] - round(V_list[i][j])) < 1e-9:
                                V_list[i][j] = int(round(V_list[i][j]))
                            else:
                                V_list[i][j] = round(V_list[i][j] * 2) / 2
                    for i in range(len(W_list)):
                        for j in range(len(W_list[i])):
                            if abs(W_list[i][j] - round(W_list[i][j])) < 1e-9:
                                W_list[i][j] = int(round(W_list[i][j]))
                            else:
                                W_list[i][j] = round(W_list[i][j] * 2) / 2
                    return U_list, V_list, W_list

    return None
# EVOLVE-BLOCK-END
