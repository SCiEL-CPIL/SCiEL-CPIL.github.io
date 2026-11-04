<<<<<<< SEARCH
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists to match the interface
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
=======
import numpy as np
import random
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    def trivial_decomposition():
        rank = trivial_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    max_time = 110.0
    
    # Search for lower rank using algebraic elimination (fixing U, V to sparse patterns)
    for r in range(trivial_rank - 1, 24, -1):
        if time.time() - start_time > max_time:
            break
            
        for attempt in range(10):
            if time.time() - start_time > max_time:
                break
                
            U = np.zeros((n*m, r), dtype=float)
            V = np.zeros((m*p, r), dtype=float)
            
            for col in range(r):
                U_col = np.random.choice([-1, 0, 1], size=n*m)
                V_col = np.random.choice([-1, 0, 1], size=m*p)
                
                if np.all(U_col == 0) or np.all(V_col == 0):
                    continue
                    
                U[:, col] = U_col
                V[:, col] = V_col
            
            num_vars = p * n * r
            num_eqs = n * m * p
            
            M = np.zeros((num_eqs, num_vars), dtype=float)
            b = np.ones(num_eqs, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(r):
                            coeff = U[i*m+j, r_idx] * V[j*p+k, r_idx]
                            if coeff != 0:
                                var_idx = (k*n + i) * r + r_idx
                                M[eq_idx, var_idx] += coeff
                        eq_idx += 1
            
            try:
                if num_vars == num_eqs:
                    W_flat = np.linalg.solve(M, b)
                else:
                    W_flat, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                    if residuals and residuals[0] > 1e-6:
                        continue
                        
            except np.linalg.LinAlgError:
                continue
                
            W_check = W_flat * 2
            if np.allclose(W_check, np.round(W_check), atol=1e-6):
                W = W_flat.reshape((p*n, r))
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    return trivial_decomposition()
>>>>>>> REPLACE
