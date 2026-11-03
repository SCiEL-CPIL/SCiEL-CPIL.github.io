"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W form a valid decomposition."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j]
                    v_val = V[j*p+k]
                    w_val = W[k*n+i]
                    prod = np.dot(u_val, v_val * w_val)
                    if abs(prod - 1.0) > tol:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U, V, W, current_rank = schoolbook()
    
    # Try to reduce rank by attempting to find a valid decomposition at lower ranks
    # Use a structured approach: try R from current_rank-1 down to some minimum
    
    best_U, best_V, best_W, best_rank = U, V, W, current_rank
    
    # For each target rank, try to find a solution using iterative refinement
    def attempt_reduction(target_rank, max_iter=1000):
        """Try to find a valid decomposition of target_rank."""
        if target_rank >= current_rank:
            return None
        
        # Initialize with random values
        random.seed(42)
        np.random.seed(42)
        
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Use gradient-free optimization: try to minimize the residual
        # residual = sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        
        def compute_residual(U, V, W):
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i*m+j]
                        v_val = V[j*p+k]
                        w_val = W[k*n+i]
                        prod = np.dot(u_val, v_val * w_val)
                        res += (prod - 1.0)**2
            return res
        
        # This is too slow for large residual computations. Let's use a different approach.
        # Instead, let's use the fact that we can fix two matrices and solve for the third.
        
        # Fix U and V, solve for W: for each (i,k), we need
        # sum_c W[k*n+i, c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
        # This is a linear system in W!
        
        # For fixed U, V:
        # For each (i,k): sum over j of... no wait.
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Group by (i,k): for fixed i,k, we have m equations (one per j) in the p*n variables W[k*n+i,:].
        # But W[k*n+i,:] has target_rank components, and we have m=5 equations.
        # So we can solve for W if the system is consistent.
        
        # Let's try: fix U and V randomly, then solve for W.
        
        for trial in range(50):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            
            # Solve for W
            W = np.zeros((p*n, target_rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # We need: for all j in range(m):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # Let A[j, c] = U[i*m+j,c] * V[j*p+k,c]
                    # Then: sum_c A[j,c] * W[k*n+i,c] = 1 for all j
                    # This is: A @ W[k*n+i,:] = ones(m)
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j] = U[i*m+j] * V[j*p+k]
                    
                    # Solve A @ x = ones(m) for x = W[k*n+i,:]
                    # Least squares solution
                    try:
                        x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        # Check if it's an exact solution
                        if np.max(np.abs(A @ x - np.ones(m))) > 1e-8:
                            valid = False
                            break
                        W[k*n+i] = x
                    except:
                        valid = False
                        break
            
            if valid:
                # Verify
                if verify(U, V, W):
                    return U, V, W
        
        return None
    
    # Try ranks from current_rank-1 down
    for target_rank in range(current_rank - 1, max(50, current_rank - 30), -1):
        if time.time() > deadline:
            break
        
        result = attempt_reduction(target_rank)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W, best_rank = U, V, W, target_rank
            # Continue trying to go lower
        else:
            # If we can't find it at this rank, stop
            break
    
    def to_lists(U, V, W):
        return {
            "U": [[U[i,j] for j in range(U.shape[1])] for i in range(U.shape[0])],
            "V": [[V[i,j] for j in range(V.shape[1])] for i in range(V.shape[0])],
            "W": [[W[i,j] for j in range(W.shape[1])] for i in range(W.shape[0])]
        }
    
    return to_lists(best_U, best_V, best_W)
# EVOLVE-BLOCK-END
