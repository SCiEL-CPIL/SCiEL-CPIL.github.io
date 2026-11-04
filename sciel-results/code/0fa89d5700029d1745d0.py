import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def compute_max_error(U, V, W):
        """Compute max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def als_update_U(U, V, W):
        """Update U given V, W. For each (i,j), solve small system."""
        new_U = np.zeros_like(U)
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                a = i * m + j
                # A[k, r] = V[j*p+k, r] * W[k*n+i, r] for k in range(p)
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                # Solve A @ x = b, where x = U[a, :]
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    new_U[a, :] = x
                except:
                    new_U[a, :] = U[a, :]
        return new_U
    
    def als_update_V(U, V, W):
        """Update V given U, W. For each (j,k), solve small system."""
        new_V = np.zeros_like(V)
        R = V.shape[1]
        for j in range(m):
            for k in range(p):
                b = j * p + k
                # A[i, r] = U[i*m+j, r] * W[k*n+i, r] for i in range(n)
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                bb = np.ones(n)
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, bb, rcond=None)
                    new_V[b, :] = x
                except:
                    new_V[b, :] = V[b, :]
        return new_V
    
    def als_update_W(U, V, W):
        """Update W given U, V. For each (k,i), solve small system."""
        new_W = np.zeros_like(W)
        R = W.shape[1]
        for k in range(p):
            for i in range(n):
                c = k * n + i
                # A[j, r] = U[i*m+j, r] * V[j*p+k, r] for j in range(m)
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                bb = np.ones(m)
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, bb, rcond=None)
                    new_W[c, :] = x
                except:
                    new_W[c, :] = W[c, :]
        return new_W
    
    def run_als(U, V, W, iterations=100):
        for _ in range(iterations):
            U = als_update_U(U, V, W)
            V = als_update_V(U, V, W)
            W = als_update_W(U, V, W)
        return U, V, W
    
    # Try ranks from 59 down to 50
    for rank in range(rank_school - 1, 49, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Multiple random restarts
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            np.random.seed(rank * 100 + restart)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Run ALS
            U, V, W = run_als(U, V, W, iterations=50)
            
            err = compute_max_error(U, V, W)
            if err < 1e-6:
                # Try rounding to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                err_int = compute_max_error(U_int, V_int, W_int)
                if err_int < 1e-9:
                    best_rank = rank
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    found = True
                    break
        
        if found:
            # Continue to see if we can go lower
            if rank <= 50:
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
