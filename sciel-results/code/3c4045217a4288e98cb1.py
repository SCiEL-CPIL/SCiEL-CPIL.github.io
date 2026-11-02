import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1
    # We want U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Start with a known good rank from history (28) and try to go lower
    # Lower bound is max(n*m, m*p, p*n) = 15? No, for 2x2x2 it is 7 > 4.
    # Let's try ranks from 20 to 29.
    
    best_result = None
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Precompute index mappings for speed
    # U_idx[i, j] = i*m + j
    # V_idx[j, k] = j*p + k
    # W_idx[k, i] = k*n + i
    
    for r in range(20, 30):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Solve for U
                # For each row i*m+j of U, we have linear equations for each (k) ??
                # No, the equation is sum over c.
                # Let's reshape U, V, W to tensors U[i, j, c], V[j, k, c], W[k, i, c]
                U_t = U.reshape(n, m, r)
                V_t = V.reshape(m, p, r)
                W_t = W.reshape(p, n, r)
                
                # For fixed V, W, U is linear.
                # Eq: sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] = 1
                # This is not linear in U if we fix V, W?
                # Wait, sum_c U[i,j,c] (V[j,k,c] W[k,i,c]).
                # Yes, for fixed V, W, it is linear in U.
                # For each (i, j), we have variables U[i, j, 0..r-1].
                # Constraints: for all k, sum_c U[i, j, c] * (V[j, k, c] * W[k, i, c]) = 1.
                # This is a system of p equations for r variables U[i, j, :].
                # We can solve least squares for each (i, j).
                
                new_U = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        # Matrix A of size p x r
                        # A[k, c] = V[j, k, c] * W[k, i, c]
                        A = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                A[k, c] = V_t[j, k, c] * W_t[k, i, c]
                        b = np.ones(p)
                        # Solve A x = b
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        new_U[i*m + j, :] = x
                        
                U = new_U
                U_t = U.reshape(n, m, r)
                
                # Solve for V
                new_V = np.zeros((m*p, r))
                for j in range(m):
                    for k in range(p):
                        # Variables V[j, k, c]
                        # Constraints: for all i, sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] = 1
                        # A[i, c] = U[i, j, c] * W[k, i, c]
                        A = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A[i, c] = U_t[i, j, c] * W_t[k, i, c]
                        b = np.ones(n)
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        new_V[j*p + k, :] = x
                        
                V = new_V
                V_t = V.reshape(m, p, r)
                
                # Solve for W
                new_W = np.zeros((p*n, r))
                for k in range(p):
                    for i in range(n):
                        # Variables W[k, i, c]
                        # Constraints: for all j, sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] = 1
                        # A[j, c] = U[i, j, c] * V[j, k, c]
                        A = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A[j, c] = U_t[i, j, c] * V_t[j, k, c]
                        b = np.ones(m)
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        new_W[k*n + i, :] = x
                        
                W = new_W
                W_t = W.reshape(p, n, r)
                
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                        error += (val - 1.0)**2
                        
            if error < 1e-8:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded
                err_rounded = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U_rounded[i*m + j, c] * V_rounded[j*p + k, c] * W_rounded[k*n + i, c]
                            err_rounded += (val - 1.0)**2
                            
                if err_rounded < 1e-9:
                    # Convert to lists of floats/ints
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    
    # If no better found, return seed (rank 30)
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
