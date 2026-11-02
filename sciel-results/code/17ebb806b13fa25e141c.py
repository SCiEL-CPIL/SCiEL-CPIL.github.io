import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute indices for non-zero tensor entries
    # T[alpha, beta, gamma] = 1 if alpha=4i+j, beta=8j+k, gamma=3k+i
    # alpha in 0..11, beta in 0..31, gamma in 0..23
    
    # For ALS updates, we need to know which equations apply to which variable.
    
    # Update U[alpha, :]:
    # alpha = 4i + j
    # Equations for k=0..p-1
    # beta = 8j + k
    # gamma = 3k + i
    # A[alpha, k, r] = V[beta, r] * W[gamma, r]
    
    # Update V[beta, :]:
    # beta = 8j + k
    # Equations for i=0..n-1
    # alpha = 4i + j
    # gamma = 3k + i
    # A[beta, i, r] = U[alpha, r] * W[gamma, r]
    
    # Update W[gamma, :]:
    # gamma = 3k + i
    # Equations for j=0..m-1
    # alpha = 4i + j
    # beta = 8j + k
    # A[gamma, j, r] = U[alpha, r] * V[beta, r]

    def solve_rank(R, max_iters=1000, tol=1e-9):
        # Initialize
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None, None, None, np.inf
            
            # Update U
            for i in range(n):
                for j in range(m):
                    alpha = i * m + j
                    # Build A matrix (p x R)
                    A = np.zeros((p, R))
                    for k in range(p):
                        beta = j * p + k
                        gamma = k * n + i
                        A[k, :] = V[beta, :] * W[gamma, :]
                    b = np.ones(p)
                    # Solve min || A u - b ||^2
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        U[alpha, :] = sol
                    except:
                        return None, None, None, np.inf
            
            # Update V
            for j in range(m):
                for k in range(p):
                    beta = j * p + k
                    A = np.zeros((n, R))
                    for i in range(n):
                        alpha = i * m + j
                        gamma = k * n + i
                        A[i, :] = U[alpha, :] * W[gamma, :]
                    b = np.ones(n)
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        V[beta, :] = sol
                    except:
                        return None, None, None, np.inf
            
            # Update W
            for k in range(p):
                for i in range(n):
                    gamma = k * n + i
                    A = np.zeros((m, R))
                    for j in range(m):
                        alpha = i * m + j
                        beta = j * p + k
                        A[j, :] = U[alpha, :] * V[beta, :]
                    b = np.ones(m)
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[gamma, :] = sol
                    except:
                        return None, None, None, np.inf
            
            # Calculate residual
            # Check a few random non-zero entries or all
            # Full check:
            # T_approx[alpha, beta, gamma] = sum_r U[alpha,r] V[beta,r] W[gamma,r]
            # We only care about the 96 non-zero entries.
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        alpha = i * m + j
                        beta = j * p + k
                        gamma = k * n + i
                        val = np.dot(U[alpha, :], V[beta, :] * W[gamma, :])
                        res += (val - 1.0)**2
            
            if res < tol:
                return U, V, W, res
        
        return None, None, None, res

    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) = 32.
    # But likely higher. Let's try a range.
    # To maximize score, we want lowest R.
    # We will try R in [40, 95]
    # If we find one, we return it.
    
    best_result = None
    best_rank = 96
    best_res = np.inf
    
    # Sort ranks to try. Maybe start from a "likely" low rank?
    # Or just linear search.
    # Given time constraints, we can't try all 40..95 with many restarts.
    # Let's try specific candidates: 48, 56, 64, 72, 80, 88, 95.
    # Or just loop from 40 to 95.
    
    ranks_to_try = list(range(40, 96))
    # Shuffle slightly? No, lower is better.
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try a few restarts for this rank
        # If R is close to 96, it's easier.
        # If R is low, it's harder.
        num_restarts = 5 if R < 60 else 2
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            U, V, W, res = solve_rank(R, max_iters=200, tol=1e-8)
            
            if U is not None and res < 1e-6:
                # Found a valid decomposition
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            if res < best_res:
                best_res = res
                best_rank = R
                best_result = (U, V, W)
        
        # If we found a very good residual but not exact, maybe continue?
        # The task requires "exactly decompose".
        # So we must return None if not exact.
        
    # If no low rank found, fall back to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
