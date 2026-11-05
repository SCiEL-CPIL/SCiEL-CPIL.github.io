import scipy
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Target tensor T[i, j, k] = 1
    # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)
    # Indexing:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Precompute indices for tensor components
    # We need to evaluate sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def error(U, V, W, R):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(R):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0) ** 2
        return err

    best_result = None
    best_rank = n * m * p # 30
    
    # Try ranks from 12 to 20
    for R in range(12, 21):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(20): # Few attempts per rank
            if time.time() > deadline:
                break
                
            # Initialize with random small integers
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            W = np.random.randint(-1, 2, (p*n, R))
            
            # Flatten for optimization
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                Ux = x[:n*m*R].reshape(n*m, R)
                Vx = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                Wx = x[n*m*R + m*p*R:].reshape(p*n, R)
                return error(Ux, Vx, Wx, R)
                
            try:
                # Use Powell method, it doesn't require gradients
                res = scipy.optimize.minimize(objective, x0, method='Powell', options={'maxiter': 50, 'xtol': 1e-8})
                if res.fun < 1e-6:
                    # Check if close to half-integers
                    Ux = res.x[:n*m*R].reshape(n*m, R)
                    Vx = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    Wx = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Round to nearest half-integer
                    U_int = np.round(Ux * 2) / 2.0
                    V_int = np.round(Vx * 2) / 2.0
                    W_int = np.round(Wx * 2) / 2.0
                    
                    # Verify exactness
                    if error(U_int, V_int, W_int, R) < 1e-9:
                        # Convert to list of lists
                        U_list = U_int.tolist()
                        V_list = V_int.tolist()
                        W_list = W_int.tolist()
                        
                        # Check if entries are integers or half-integers
                        # They are by construction
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass
                
    # If no better found, return seed
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
