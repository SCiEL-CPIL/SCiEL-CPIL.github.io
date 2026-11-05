"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, max_iter=2000, seed=42):
    """Try to find a decomposition of rank r using iterative optimization."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, r)
    V = rng.randn(m*p, r)
    W = rng.randn(p*n, r)
    
    # Precompute indices
    idx_U = [(i*m+j, j, i) for i in range(n) for j in range(m)]  # (U_idx, j, i)
    idx_V = [(j*p+k, j, k) for j in range(m) for k in range(p)]  # (V_idx, j, k)
    idx_W = [(k*n+i, k, i) for k in range(p) for i in range(n)]  # (W_idx, k, i)
    
    # For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # We can do alternating minimization: fix two, solve for third linearly
    
    for it in range(max_iter):
        # Update W: for each (k,i) and each c, W[k*n+i,c] appears in equations for all j
        # Equation for (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Fix U and V, solve for W row by row
        for k in range(p):
            for i in range(n):
                wi = k*n+i
                # For each c, coefficient of W[wi,c] in equation (i,j,k) is U[i*m+j,c]*V[j*p+k,c]
                # We have m equations (one for each j) for this (k,i) pair
                # A[wi,c] = U[i*m+j,c] * V[j*p+k,c] for j=0..m-1
                # We want: for each j, sum_c A[j,c] * W[wi,c] = 1
                # This is a linear system: A @ W[wi,:] = ones(m)
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                # Solve A @ w = 1
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[wi, :] = w_sol
                except:
                    pass
        
        # Update V: for each (j,k) and each c
        for j in range(m):
            for k in range(p):
                vj = j*p+k
                # For each i, equation (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # Fix U and W, solve for V[j*p+k,:]
                # Coefficient of V[vj,c] in eq (i,j,k) is U[i*m+j,c]*W[k*n+i,c]
                # n equations for this (j,k)
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[vj, :] = v_sol
                except:
                    pass
        
        # Update U: for each (i,j) and each c
        for i in range(n):
            for j in range(m):
                ui = i*m+j
                # For each k, equation (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # Fix V and W, solve for U[i*m+j,:]
                # p equations for this (i,j)
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[ui, :] = u_sol
                except:
                    pass
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0)**2
        if err < 1e-12:
            # Convert to list of lists with rounding
            U_list = [[round(float(x), 10) for x in row] for row in U]
            V_list = [[round(float(x), 10) for x in row] for row in V]
            W_list = [[round(float(x), 10) for x in row] for row in W]
            # Final verification
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try ranks from schoolbook-1 down to some minimum
    # Theoretical lower bound: at least max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But realistically for matrix multiplication, rank is at least n*m*p / something
    # Let's try a range of ranks
    for r in range(schoolbook_rank - 1, 17, -1):
        if time.time() > deadline:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            result = _try_rank(r, n, m, p, max_iter=500, seed=seed + r*100)
            if result is not None:
                return result
    
    # Fallback: schoolbook
    rank = schoolbook_rank
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
