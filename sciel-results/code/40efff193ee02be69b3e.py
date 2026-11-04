import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Indices
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Precompute the linear system matrices for ALS
    # Equation: sum_c U[(i*m+j), c] * V[(j*p+k), c] * W[(k*n+i), c] = 1
    
    # We will try ranks from 72 down to some lower bound
    # Start with 72, then 70, 68, etc.
    # Actually, let's just try to find a good rank.
    
    best_rank = n * m * p
    best_solution = None
    
    # Time management
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # Try ranks
    # We know 72 works.
    # Let's try to find a smaller one.
    # Common ranks to try: 72, 70, 68, 64, 60
    
    for r in range(72, 40, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for this rank
        found = False
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS iterations
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                    
                # Solve for U
                # Matrix A_U of size (n*m*p) x (n*m*r)
                # Row index: (i, j, k) -> i*m*p + j*p + k ? No, we need to map to equation index.
                # Let's just build the system explicitly for speed since dimensions are small.
                
                # System for U:
                # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # This is not a standard linear system for U because the coefficient depends on k.
                # Wait, for a fixed U update, we fix V and W.
                # The equation is linear in U.
                # Let's vectorize U as a vector of length n*m*r.
                # Unknown U_vec.
                # Equation for (i, j, k):
                # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # Let coeff[(i,j,k), (i,j,c)] = V[j*p+k, c] * W[k*n+i, c]
                
                # We can build the matrix M_U of size (n*m*p) x (n*m*r)
                # But n*m*p = 72. n*m*r is at most 9*72 = 648.
                # This is small.
                
                M_U = np.zeros((n*m*p, n*m*r))
                b_U = np.ones(n*m*p)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for c in range(r):
                                col_idx = (i * m + j) * r + c
                                M_U[eq_idx, col_idx] = V[j * p + k, c] * W[k * n + i, c]
                
                # Solve M_U U_vec = b_U
                U_vec, _, _, _ = np.linalg.lstsq(M_U, b_U, rcond=None)
                U = U_vec.reshape(n * m, r)
                
                # Solve for V
                M_V = np.zeros((n*m*p, m*p*r))
                b_V = np.ones(n*m*p)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for c in range(r):
                                col_idx = (j * p + k) * r + c
                                M_V[eq_idx, col_idx] = U[i * m + j, c] * W[k * n + i, c]
                
                V_vec, _, _, _ = np.linalg.lstsq(M_V, b_V, rcond=None)
                V = V_vec.reshape(m * p, r)
                
                # Solve for W
                M_W = np.zeros((n*m*p, p*n*r))
                b_W = np.ones(n*m*p)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for c in range(r):
                                col_idx = (k * n + i) * r + c
                                M_W[eq_idx, col_idx] = U[i * m + j, c] * V[j * p + k, c]
                
                W_vec, _, _, _ = np.linalg.lstsq(M_W, b_W, rcond=None)
                W = W_vec.reshape(p * n, r)
                
                # Check error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            error += (val - 1) ** 2
                
                if error < 1e-6:
                    found = True
                    best_rank = r
                    best_solution = (U, V, W)
                    break
                    
            if found:
                break
                
        if found:
            break
            
    if best_solution is None:
        # Fallback to schoolbook
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
