"""Matrix multiplication 3x3x6 tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the decomposition."""
    rank = U.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                error += (val - 1.0) ** 2
    return error

def _compute_gradients(U, V, W, n, m, p):
    """Compute gradients of the error with respect to U, V, W."""
    rank = U.shape[1]
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Current value
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # Error term: 2 * (val - 1)
                err_term = 2.0 * (val - 1.0)
                
                for c in range(rank):
                    dU[i*m+j, c] += err_term * V[j*p+k, c] * W[k*n+i, c]
                    dV[j*p+k, c] += err_term * U[i*m+j, c] * W[k*n+i, c]
                    dW[k*n+i, c] += err_term * U[i*m+j, c] * V[j*p+k, c]
    
    return dU, dV, dW

def _optimize_decomposition(U, V, W, n, m, p, max_iter=2000, tol=1e-10, lr=0.01, time_limit=None, start_time=None):
    """Optimize the decomposition using gradient descent with adaptive learning rate."""
    if start_time is None:
        start_time = time.time()
    
    current_error = _compute_error(U, V, W, n, m, p)
    best_error = current_error
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Adaptive learning rate
    effective_lr = lr
    prev_error = current_error
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        dU, dV, dW = _compute_gradients(U, V, W, n, m, p)
        
        # Normalize gradients
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm < 1e-12:
            break
        
        # Update
        U -= effective_lr * dU
        V -= effective_lr * dV
        W -= effective_lr * dW
        
        new_error = _compute_error(U, V, W, n, m, p)
        
        # Adaptive learning rate
        if new_error > prev_error * 1.5:
            effective_lr *= 0.5
            # Revert
            U += effective_lr * 2.0 * dU  # Approximate revert
            V += effective_lr * 2.0 * dV
            W += effective_lr * 2.0 * dW
            new_error = _compute_error(U, V, W, n, m, p)
        else:
            # Slowly increase if making progress
            if new_error < prev_error * 0.95:
                effective_lr = min(effective_lr * 1.05, 0.1)
        
        prev_error = new_error
        
        if new_error < best_error:
            best_error = new_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_error < tol:
            break
    
    return best_U, best_V, best_W, best_error

def _random_search(n, m, p, target_rank, num_restarts=20, time_limit=30, start_time=None):
    """Random search with multiple restarts and gradient optimization."""
    if start_time is None:
        start_time = time.time()
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        np.random.seed(12345 + restart * 7919 + target_rank * 100003)
        
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.3
        V = np.random.randn(m * p, target_rank) * 0.3
        W = np.random.randn(p * n, target_rank) * 0.3
        
        # Optimize
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining < 1:
            break
        
        U_opt, V_opt, W_opt, error = _optimize_decomposition(
            U, V, W, n, m, p,
            max_iter=500,
            tol=1e-12,
            lr=0.05,
            time_limit=time_remaining,
            start_time=start_time
        )
        
        if error < best_error:
            best_error = error
            best_result = (U_opt, V_opt, W_opt)
        
        if best_error < 1e-10:
            break
    
    return best_result, best_error

def _try_structured_decomposition(n, m, p):
    """
    Try a structured approach: exploit the block structure.
    
    For 3x3x6, view B as [B1 | B2] where B1, B2 are 3x3.
    C = [A*B1 | A*B2]
    
    We can try to share some computations between the two halves.
    """
    # This is a placeholder for more sophisticated structured approaches
    # For now, return None to indicate no improvement found
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = result
    
    # Deadline guard: 150s wall-clock, use 145s as safety margin
    deadline = time.time() + 145
    start_time = time.time()
    
    # Try various ranks from low to high
    # Theoretical lower bound is around n*p = 18, but likely higher
    # Try ranks from 30 to 53
    for target_rank in range(30, 54):
        if time.time() > deadline - 5:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 3:
            break
        
        # Allocate time based on rank
        # Lower ranks get more time since they're harder
        time_for_rank = min(time_remaining * 0.4, 20)
        
        search_result, error = _random_search(
            n, m, p, target_rank,
            num_restarts=30,
            time_limit=time_for_rank,
            start_time=start_time
        )
        
        if search_result is not None and error < 1e-8:
            # Verify
            U_list = search_result[0].tolist()
            V_list = search_result[1].tolist()
            W_list = search_result[2].tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found the lowest rank, stop
        
        # If we found a good solution for this rank, we can stop
        if search_result is not None and error < 1e-6:
            # This rank works, try to go lower
            continue
    
    # Try structured approach if time permits
    if time.time() < deadline - 10:
        structured = _try_structured_decomposition(n, m, p)
        if structured is not None:
            rank = len(structured["U"][0])
            if rank < best_rank:
                if _verify_decomposition(structured["U"], structured["V"], structured["W"], n, m, p):
                    best_result = structured
                    best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
