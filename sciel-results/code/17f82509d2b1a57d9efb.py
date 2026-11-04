import numpy as np
import time
import random

def construct():
    n, m, p = 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def schoolbook_solution():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Vectorized verification of tensor decomposition."""
        rank = U.shape[1]
        # For each (i,j,k), check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(np.dot(U[u_idx, :], V[v_idx, :]), W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, tol=1e-9):
        """Faster verification using matrix operations."""
        rank = U.shape[1]
        # Build all required products
        # For each i,j: U[i*m+j, :] is a row vector
        # For each j,k: V[j*p+k, :] is a row vector  
        # For each k,i: W[k*n+i, :] is a row vector
        # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        # Precompute outer products
        # For fixed (i,j), define M_ij[k,r] = V[j*p+k,r] * W[k*n+i,r]
        # Then we need: sum_r U[i*m+j,r] * M_ij[k,r] = 1 for all k
        
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j, :]
                # Build M matrix of shape (p, rank)
                M = np.zeros((p, rank))
                for k in range(p):
                    v_row = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    M[k, :] = v_row * w_row
                # Check: M @ u_row should be ones
                result = M @ u_row
                if np.any(np.abs(result - 1.0) > tol):
                    return False
        return True
    
    def als_decomposition(target_rank, max_iters=100, tol=1e-10, time_budget=10):
        """Alternating Least Squares for tensor decomposition.
        
        Returns (U, V, W) if converged, else None.
        """
        start_t = time.time()
        
        # Random initialization
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        prev_error = float('inf')
        
        for it in range(max_iters):
            if time.time() - start_t > time_budget:
                return None
            
            # Update U: fix V and W
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # Build A matrix: A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        v_row = V[j * p + k, :]
                        w_row = W[k * n + i, :]
                        A[k, :] = v_row * w_row
                    b = np.ones(p)
                    try:
                        U[u_idx, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U[u_idx, :] = np.random.randn(target_rank) * 0.1
            
            # Update V: fix U and W
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    # Build A matrix: A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        u_row = U[i * m + j, :]
                        w_row = W[k * n + i, :]
                        A[i, :] = u_row * w_row
                    b = np.ones(n)
                    try:
                        V[v_idx, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V[v_idx, :] = np.random.randn(target_rank) * 0.1
            
            # Update W: fix U and V
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    # Build A matrix: A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        u_row = U[i * m + j, :]
                        v_row = V[j * p + k, :]
                        A[j, :] = u_row * v_row
                    b = np.ones(m)
                    try:
                        W[w_idx, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W[w_idx, :] = np.random.randn(target_rank) * 0.1
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(np.dot(U[u_idx, :], V[v_idx, :]), W[w_idx, :])
                        error += (val - 1.0) ** 2
            error = np.sqrt(error)
            
            if error < tol:
                return U, V, W
            
            if abs(prev_error - error) < 1e-12:
                if error < 1e-6:
                    return U, V, W
                break
            
            prev_error = error
        
        # Final check
        if verify_decomposition_fast(U, V, W, tol=1e-8):
            return U, V, W
        
        return None
    
    # Start with schoolbook as fallback
    best_solution = schoolbook_solution()
    best_rank = n * m * p
    
    # Lower bound for rank
    min_possible_rank = max(n * m, m * p, p * n)  # = 36
    
    # Try ranks from lower bound upward
    # Be strategic: try a few key ranks first
    ranks_to_try = []
    for r in range(min_possible_rank, min(best_rank, 60)):
        ranks_to_try.append(r)
    
    random.shuffle(ranks_to_try)
    # Put lower ranks first
    ranks_to_try.sort()
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Multiple restarts for this rank
        max_restarts = 10 if target_rank <= 40 else 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break
            
            result = als_decomposition(target_rank, max_iters=50, tol=1e-10, time_budget=5)
            if result is not None:
                U, V, W = result
                if verify_decomposition_fast(U, V, W, tol=1e-8):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        found = True
                        break
        
        if found:
            # Try to go even lower
            continue
    
    return best_solution
