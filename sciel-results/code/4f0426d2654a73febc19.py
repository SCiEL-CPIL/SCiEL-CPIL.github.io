"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    def verify_exact(U, V, W):
        """Check if the decomposition is exact with integer/half-integer entries."""
        rank = len(U[0])
        # Check entries are integer or half-integer
        for mat_name, mat in [('U', U), ('V', V), ('W', W)]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        # Check tensor equation
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank(r, max_iters=200):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Precompute index mappings
        u_idx = [(i, j) for i in range(n) for j in range(m)]
        v_idx = [(j, k) for j in range(m) for k in range(p)]
        w_idx = [(k, i) for k in range(p) for i in range(n)]
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over k,c = 1
                    # For fixed (i,j), we need: sum_c U[i*m+j,c] * (sum_k V[j*p+k,c] * W[k*n+i,c]) = 1
                    # But this couples all k's. Let's do it differently.
                    pass
            
            # Better: use the full tensor equation
            # Residual for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1
            # Update U[i*m+j, :] to minimize sum_{k} (sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1)^2
            # This is a linear system in U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # Build system: A @ x = b where A[c, c'] = sum_k V[j*p+k,c] * W[k*n+i,c']
                    # b[c] = sum_k V[j*p+k,c] * 1 ... no
                    # We want: min_x sum_k (sum_c x[c] * V[j*p+k,c] * W[k*n+i,c] - 1)^2
                    # = min_x ||M @ x - 1_k||^2 where M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    M = np.array([[V[j*p+k, c] * W[k*n+i, c] for c in range(r)] for k in range(p)])
                    b = np.ones(p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Update V similarly
            for j in range(m):
                for k in range(p):
                    # min_y sum_i (sum_c U[i*m+j,c] * y[c] * W[k*n+i,c] - 1)^2
                    M = np.array([[U[i*m+j, c] * W[k*n+i, c] for c in range(r)] for i in range(n)])
                    b = np.ones(n)
                    try:
                        y, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        V[j*p+k, :] = y
                    except:
                        pass
            
            # Update W similarly
            for k in range(p):
                for i in range(n):
                    # min_z sum_j (sum_c U[i*m+j,c] * V[j*p+k,c] * z[c] - 1)^2
                    M = np.array([[U[i*m+j, c] * V[j*p+k, c] for c in range(r)] for j in range(m)])
                    b = np.ones(m)
                    try:
                        z, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        W[k*n+i, :] = z
                    except:
                        pass
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(total - 1.0))
            
            if max_err < 1e-10:
                # Found a solution, try to round
                U_rounded = [[round(val * 2) / 2.0 for val in U[row]] for row in range(n*m)]
                V_rounded = [[round(val * 2) / 2.0 for val in V[row]] for row in range(m*p)]
                W_rounded = [[round(val * 2) / 2.0 for val in W[row]] for row in range(p*n)]
                if verify_exact(U_rounded, V_rounded, W_rounded):
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                # Try different rounding
                for offset in [0, 0.25, 0.5, 0.75]:
                    U_r = [[round((val + offset) * 2) / 2.0 for val in U[row]] for row in range(n*m)]
                    V_r = [[round((val + offset) * 2) / 2.0 for val in V[row]] for row in range(m*p)]
                    W_r = [[round((val + offset) * 2) / 2.0 for val in W[row]] for row in range(p*n)]
                    if verify_exact(U_r, V_r, W_r):
                        return {"U": U_r, "V": V_r, "W": W_r}
        
        return None
    
    # Try ranks from 25 to 49
    best_result = None
    for r in range(25, 50):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        return schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
