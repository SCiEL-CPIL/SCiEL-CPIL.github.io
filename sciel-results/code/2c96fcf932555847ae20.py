import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def compute_error(U, V, W):
        """Compute the max error of the decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def alternating_minimization(U, V, W, target_rank, max_iters=50, tol=1e-10, time_limit=None):
        """
        Alternating minimization for tensor decomposition using least squares.
        """
        if time_limit is not None and time.time() > time_limit:
            return U, V, W
        
        for iteration in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Solve for W: for fixed U, V, find W minimizing sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # This is a linear system in W.
            # Let's construct the linear system.
            # For each (i,k) pair, W[k*n+i] appears in the sum for all j.
            # The gradient condition: for each (i,k,r), 
            # sum_j U[i*m+j,r]*V[j*p+k,r]*(sum_r' U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'] - 1) = 0
            # This is still coupled across r.
            
            # Alternative approach: use a vectorized least squares.
            # Let's flatten the problem.
            # We want to find W such that for all (i,j,k):
            # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a linear system in W if we treat each r separately, but they're coupled.
            # 
            # Let's use a different approach: treat it as a matrix equation.
            # Let A be a matrix where A[(i,j,k), (i',k',r)] = U[i*m+j,r]*V[j*p+k,r] if i'=i, k'=k, else 0
            # Then we want A * W_flat = 1_vector
            # This is a huge system. Let's use iterative methods.
            
            # For now, use gradient descent with better step size.
            current_vals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        current_vals[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
            
            # Gradient for W
            grad_W = np.zeros_like(W)
            for r in range(target_rank):
                for i in range(n):
                    for k in range(p):
                        s = 0.0
                        for j in range(m):
                            s += U[i * m + j, r] * V[j * p + k, r] * (current_vals[i, j, k] - 1)
                        grad_W[k * n + i, r] = 2 * s
            
            if np.max(np.abs(grad_W)) < 1e-15:
                continue
                
            # Use a more sophisticated step size
            grad_norm = np.linalg.norm(grad_W)
            if grad_norm < 1e-15:
                continue
                
            # Line search
            best_step = 0.1
            best_err = compute_error(U, V, W)
            for step in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]:
                W_new = W - step * grad_W
                err = compute_error(U, V, W_new)
                if err < best_err:
                    best_err = err
                    best_step = step
                    W = W_new
                    break
            
            # Similarly for U
            current_vals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        current_vals[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        
            grad_U = np.zeros_like(U)
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        s = 0.0
                        for k in range(p):
                            s += V[j * p + k, r] * W[k * n + i, r] * (current_vals[i, j, k] - 1)
                        grad_U[i * m + j, r] = 2 * s
                        
            if np.max(np.abs(grad_U)) > 1e-15:
                best_err = compute_error(U, V, W)
                for step in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]:
                    U_new = U - step * grad_U
                    err = compute_error(U_new, V, W)
                    if err < best_err:
                        best_err = err
                        U = U_new
                        break
                        
            # Similarly for V
            current_vals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        current_vals[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        
            grad_V = np.zeros_like(V)
            for r in range(target_rank):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for i in range(n):
                            s += U[i * m + j, r] * W[k * n + i, r] * (current_vals[i, j, k] - 1)
                        grad_V[j * p + k, r] = 2 * s
                        
            if np.max(np.abs(grad_V)) > 1e-15:
                best_err = compute_error(U, V, W)
                for step in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]:
                    V_new = V - step * grad_V
                    err = compute_error(U, V_new, W)
                    if err < best_err:
                        best_err = err
                        V = V_new
                        break
            
            if compute_error(U, V, W) < tol:
                break
        
        return U, V, W
    
    # Strategy: Use alternating minimization with multiple restarts at various ranks
    # Try ranks below schoolbook
    target_ranks = [60, 58, 56, 54, 52, 50, 48]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        found = False
        num_attempts = 20
        for attempt in range(num_attempts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(attempt * 7919 + target_rank * 1000)
            
            # Initialize with structured random values
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Run alternating minimization
            U, V, W = alternating_minimization(U, V, W, target_rank, max_iters=30, tol=1e-10, time_limit=deadline - 15)
            
            # Check if we found a solution
            if check_decomposition(U, V, W, tol=1e-8):
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_decomposition(U_r, V_r, W_r, tol=1e-8):
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                else:
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                found = True
                break
        
        if found:
            break
    
    # If still at schoolbook rank, try more aggressive search
    if best_rank == rank_schoolbook and time.time() < deadline - 10:
        for target_rank in [64, 62, 60, 58]:
            if time.time() > deadline - 10:
                break
            
            for attempt in range(25):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(attempt * 5000 + target_rank * 3000)
                scale = 1.0 / np.sqrt(target_rank)
                U = np.random.randn(n * m, target_rank) * scale
                V = np.random.randn(m * p, target_rank) * scale
                W = np.random.randn(p * n, target_rank) * scale
                
                U, V, W = alternating_minimization(U, V, W, target_rank, max_iters=40, tol=1e-10, time_limit=deadline - 10)
                
                if check_decomposition(U, V, W, tol=1e-8):
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break
            
            if best_rank < rank_schoolbook:
                break
    
    # Return the best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
