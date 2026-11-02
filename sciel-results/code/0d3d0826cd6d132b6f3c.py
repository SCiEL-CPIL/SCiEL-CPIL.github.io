import numpy as np
import time
import random
import scipy.optimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Target rank 17
    R = 17
    
    # Try a few random initializations
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random binary vectors
        U = np.random.randint(0, 2, (n * m, R)).astype(float)
        V = np.random.randint(0, 2, (m * p, R)).astype(float)
        W = np.random.randint(0, 2, (p * n, R)).astype(float)
        
        # Define the residual function for least_squares
        # We flatten U, V, W into a single vector x
        def residual(x):
            U_flat = x[:n*m*R].reshape(n*m, R)
            V_flat = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W_flat = x[n*m*R + m*p*R :].reshape(p*n, R)
            
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute sum over c
                        s = 0
                        for c in range(R):
                            s += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                        res[idx] = s - 1
                        idx += 1
            return res

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = scipy.optimize.least_squares(residual, x0, method='trf', max_nfev=1000)
            if result.success or result.cost < 1e-6:
                U_sol = result.x[:n*m*R].reshape(n*m, R)
                V_sol = result.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_sol = result.x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Round to half-integers
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                # Verify
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(R):
                                val += W_rounded[k*n+i, c] * U_rounded[i*m+j, c] * V_rounded[j*p+k, c]
                            error += (val - 1)**2
                            
                if error < 1e-6:
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        except Exception:
            pass

    # Fallback to schoolbook rank 24
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
