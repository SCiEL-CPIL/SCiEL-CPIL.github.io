"""Improved portfolio: faster ALS + structured initialization for 2x5x5 matrix multiplication tensor."""

import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = 2, 5, 5
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    n_constraints = n * m * p  # 50
    
    # Build schoolbook decomposition (rank 50)
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    best_solution = build_schoolbook()
    best_rank = n * m * p
    
    # Precompute index mappings for speed
    # For each constraint (i,j,k) -> row index
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    
    # Vectorized ALS: solve for U given V, W
    # For each (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
    # This is a linear system: for fixed (i,j), the unknowns are U[i*m+j, :] (size R)
    # and the equations are for each k: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
    # So for each (i,j), we have a p x R system.
    
    def solve_U(V, W, R):
        """Solve for U given V and W. U shape (nm, R)."""
        U = np.zeros((nm, R))
        for i in range(n):
            for j in range(m):
                # Build p x R matrix A where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                # Solve A @ U[i*m+j, :]^T = b  =>  U[i*m+j, :] = (A^T A)^{-1} A^T b
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    U[i*m+j, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
        return U
    
    def solve_V(U, W, R):
        """Solve for V given U and W. V shape (mp, R)."""
        V = np.zeros((mp, R))
        for j in range(m):
            for k in range(p):
                # Build n x R matrix A where A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    V[j*p+k, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
        return V
    
    def solve_W(U, V, R):
        """Solve for W given U and V. W shape (pn, R)."""
        W = np.zeros((pn, R))
        for k in range(p):
            for i in range(n):
                # Build m x R matrix A where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    W[k*n+i, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
        return W
    
    def check_error(U, V, W):
        """Compute max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_rank(R, n_restarts=3, max_iter=200, tol=1e-9):
        """Try to find a valid decomposition of rank R."""
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Different initialization strategies
            if restart == 0:
                # Near-schoolbook: start with schoolbook and randomly combine columns
                U0, V0, W0 = build_schoolbook()
                U0 = np.array(U0)
                V0 = np.array(V0)
                W0 = np.array(W0)
                
                # Randomly project down to rank R
                np.random.seed(42 + restart)
                if R < n * m * p:
                    # Select R columns or use random linear combination
                    idx = np.random.choice(n*m*p, size=R, replace=False)
                    U = U0[:, idx]
                    V = V0[:, idx]
                    W = W0[:, idx]
                else:
                    U = U0.copy()
                    V = V0.copy()
                    W = W0.copy()
            elif restart == 1:
                # Random small initialization
                np.random.seed(123 + restart)
                U = np.random.randn(nm, R) * 0.1
                V = np.random.randn(mp, R) * 0.1
                W = np.random.randn(pn, R) * 0.1
            else:
                # Random with some structure
                np.random.seed(456 + restart)
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
                # Make some entries zero to encourage sparsity
                U[np.random.rand(nm, R) < 0.3] = 0
                V[np.random.rand(mp, R) < 0.3] = 0
                W[np.random.rand(pn, R) < 0.3] = 0
            
            # Alternating least squares
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                U = solve_U(V, W, R)
                V = solve_V(U, W, R)
                W = solve_W(U, V, R)
                
                err = check_error(U, V, W)
                
                if err < tol:
                    return U.tolist(), V.tolist(), W.tolist()
                
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
        
        return None
    
    # Try ranks from just below 50 downward
    # Focus on ranks that might be achievable
    for R in range(best_rank - 1, max(best_rank - 15, 35), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(R)
        if result is not None:
            best_rank = R
            best_solution = result
            # Continue trying lower ranks
            continue
        else:
            # If we can't find rank R, try a few more lower ranks but with less effort
            # Actually, let's just stop and return what we have
            # But maybe try R-1, R-2 with fewer restarts
            for R2 in range(R-1, max(R-3, 35), -1):
                if time.time() > deadline - 5:
                    break
                result2 = try_rank(R2, n_restarts=1, max_iter=100)
                if result2 is not None:
                    best_rank = R2
                    best_solution = result2
                    break
            break
    
    return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
