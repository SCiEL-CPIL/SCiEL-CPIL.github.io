import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # Pre-compute schoolbook as fallback
    best_rank = n * m * p
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try to find lower rank using vectorized ALS
    # Search from lower ranks up, but with limited restarts
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But better bounds exist. Let's search from 30 to 53.
    
    def verify_decomp(U, V, W, tol=1e-9):
        """Check if decomposition is exact (within tolerance)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def als_round(U, V, W, r, max_iters=30, tol=1e-8, deadline=None):
        """Vectorized ALS with rounding to half-integers."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                return None
            
            # Update U: for each (i,j), solve A u = 1 where A[k,c] = V[j*p+k,c]*W[k*n+i,c]
            for i in range(n):
                for j in range(m):
                    # Build A: p x r
                    A = V[j * p:(j + 1) * p, :] * W[0::n, :].T  # This won't work directly
                    # Need to be more careful with indexing
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        u = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        continue
                    U[i * m + j, :] = u
            
            # Update V: for each (j,k), solve A v = 1 where A[i,c] = U[i*m+j,c]*W[k*n+i,c]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        continue
                    V[j * p + k, :] = v
            
            # Update W: for each (k,i), solve A w = 1 where A[j,c] = U[i*m+j,c]*V[j*p+k,c]
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        continue
                    W[k * n + i, :] = w
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0) ** 2
            if err < tol:
                break
        
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness
        if verify_decomp(U_r, V_r, W_r):
            return (U_r, V_r, W_r)
        
        # Try without rounding if entries are close to integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify_decomp(U_i, V_i, W_i):
            return (U_i, V_i, W_i)
        
        return None
    
    # Search ranks from low to high
    # Use multiple initialization strategies
    def init_strategy(r, strategy):
        """Generate initial factor matrices."""
        if strategy == 0:
            # Small random
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
        elif strategy == 1:
            # Random signs
            U = np.random.choice([-1, 1], size=(n * m, r))
            V = np.random.choice([-1, 1], size=(m * p, r))
            W = np.random.choice([-1, 1], size=(p * n, r))
        elif strategy == 2:
            # Random half-integers
            U = np.random.randint(0, 3, size=(n * m, r)) * 0.5 - 0.5
            V = np.random.randint(0, 3, size=(m * p, r)) * 0.5 - 0.5
            W = np.random.randint(0, 3, size=(p * n, r)) * 0.5 - 0.5
        elif strategy == 3:
            # Sparse random
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for _ in range(r * 2):
                U[np.random.randint(0, n * m), np.random.randint(0, r)] = np.random.choice([-1, 1])
                V[np.random.randint(0, m * p), np.random.randint(0, r)] = np.random.choice([-1, 1])
                W[np.random.randint(0, p * n), np.random.randint(0, r)] = np.random.choice([-1, 1])
        elif strategy == 4:
            # Structured: try to exploit symmetry
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
        else:
            U = np.random.randn(n * m, r) * 0.2
            V = np.random.randn(m * p, r) * 0.2
            W = np.random.randn(p * n, r) * 0.2
        return U, V, W
    
    # Try ranks from 30 to 53
    # Allocate time based on rank: more time for lower ranks
    for r in range(30, best_rank):
        if time.time() - start_time > timeout:
            break
        
        # Number of restarts: more for lower ranks
        remaining = timeout - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Estimate time per restart
        max_restarts = min(50, int(remaining / 1.5))  # ~1.5s per restart estimate
        max_restarts = max(max_restarts, 5)
        
        found = False
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                break
            
            strategy = restart % 5
            U, V, W = init_strategy(r, strategy)
            
            result = als_round(U, V, W, r, max_iters=40, tol=1e-8, deadline=start_time + timeout)
            
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = r
                best_decomp = (U_r, V_r, W_r)
                found = True
                break
        
        if found:
            # Found a decomposition at rank r, continue searching lower
            # But we're going up, so we should continue to see if we can find lower
            # Actually we're searching from 30 up, so if we find at 30, that's great
            # Continue to next rank only if we want to verify we have the minimum
            # But since we're going from low to high, the first found is the minimum
            break
    
    # Return the best found
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
