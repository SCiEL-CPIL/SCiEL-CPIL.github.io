"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Build the target tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Start with schoolbook as fallback
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find lower rank decompositions
    # Use random initialization and check if we can solve the linear system
    # For a given rank r, we have r unknowns per entry in U, V, W
    # The constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Strategy: try various ranks, use random init, then do gradient descent
    # to minimize sum of squared errors
    
    def compute_error(U, V, W):
        """Compute the sum of squared errors for the decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k]) * 0  # placeholder
                    # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # This is not a simple dot product of two vectors
                    # It's a trilinear form
                    pass
        # Better: compute efficiently
        # For each c, the contribution is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # We want the sum over c to equal the all-ones tensor
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    total_err += (s - 1.0) ** 2
        return total_err
    
    # More efficient error computation using numpy
    def compute_error_fast(U, V, W):
        """Compute sum of squared errors efficiently."""
        r = U.shape[1]
        err = 0.0
        # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # We can precompute for efficiency
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j]
                for k in range(p):
                    v_row = V[j*p+k]
                    w_row = W[k*n+i]
                    # sum_c u_row[c] * v_row[c] * w_row[c]
                    s = np.sum(u_row * v_row * w_row)
                    err += (s - 1.0) ** 2
        return err
    
    # Gradient computation
    def compute_gradients(U, V, W):
        """Compute gradients of the error with respect to U, V, W."""
        r = U.shape[1]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err_term = 2.0 * (s - 1.0)
                    dU[i*m+j] += err_term * V[j*p+k] * W[k*n+i]
                    dV[j*p+k] += err_term * U[i*m+j] * W[k*n+i]
                    dW[k*n+i] += err_term * U[i*m+j] * V[j*p+k]
        return dU, dV, dW
    
    # Try ranks from just below schoolbook down to some minimum
    # For 3x5x6, known results suggest rank might be around 70-85
    # Let's try a range of ranks with random initialization
    
    for r in range(min(rank_schoolbook - 1, 85), 50, -1):
        if time.time() > deadline - 30:
            break
        
        best_err_for_r = float('inf')
        best_U_r, best_V_r, best_W_r = None, None, None
        
        # Multiple random restarts
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Gradient descent with adaptive learning rate
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                dU, dV, dW = compute_gradients(U, V, W)
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if step % 50 == 49:
                    err = compute_error_fast(U, V, W)
                    if err < best_err_for_r:
                        best_err_for_r = err
                        best_U_r = U.copy()
                        best_V_r = V.copy()
                        best_W_r = W.copy()
                    if err < 1e-6:
                        break
                    # Reduce learning rate if error is not decreasing
                    lr *= 0.5
            
            err = compute_error_fast(U, V, W)
            if err < best_err_for_r:
                best_err_for_r = err
                best_U_r = U.copy()
                best_V_r = V.copy()
                best_W_r = W.copy()
        
        if best_err_for_r < 1e-6 and best_U_r is not None:
            best_rank = r
            best_U, best_V, best_W = best_U_r, best_V_r, best_W_r
            # Check if we can go lower
            continue
        else:
            # If we couldn't achieve low error at this rank, stop
            if best_err_for_r > 1e-3:
                break
    
    # Convert to lists of lists with integer/half-integer entries if possible
    # Round to nearest half-integer
    def to_half_int(arr):
        """Round to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    # Check if the rounded version still works
    U_rounded = to_half_int(best_U)
    V_rounded = to_half_int(best_V)
    W_rounded = to_half_int(best_W)
    
    err_rounded = compute_error_fast(U_rounded, V_rounded, W_rounded)
    if err_rounded < 1e-9:
        U_final, V_final, W_final = U_rounded, V_rounded, W_rounded
    else:
        # Use the unrounded version if rounding breaks it
        # But we need integer/half-integer entries...
        # Try to find a better solution or use schoolbook
        if best_rank < rank_schoolbook:
            # Try harder to get exact half-integer solution
            # For now, fall back to schoolbook if we can't get exact
            U_final, V_final, W_final = U_fb, V_fb, W_fb
        else:
            U_final, V_final, W_final = U_fb, V_fb, W_fb
    
    # Convert numpy arrays to lists of lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    # Ensure all values are integers or half-integers
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx, val in enumerate(row):
                # Round to nearest half-integer
                row[idx] = round(val * 2) / 2.0
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
