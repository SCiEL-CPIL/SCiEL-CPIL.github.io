import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Target tensor T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Precompute indices for the tensor entries
    # i, j, k
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
    
    # Function to calculate error given U, V, W
    def calc_error(U, V, W):
        err = 0.0
        for i, j, k in indices:
            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
            # val is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            err += (val - 1.0)**2
        return err

    # Try ranks from low to high
    # Schoolbook is 36. We hope for < 36.
    # Known lower bound for 2x3 by 3x6? 
    # For 2x2 by 2x2, rank 7.
    # For 2x3 by 3x3, rank 10 is possible? (Schoolbook 18).
    # Let's try ranks 20 to 35.
    
    best_rank = 36
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U_school = np.zeros((n*m, 36))
    V_school = np.zeros((m*p, 36))
    W_school = np.zeros((p*n, 36))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    # If we don't find anything better, return schoolbook
    if time.time() > deadline - 10:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Search for lower rank
    for R in range(20, 36):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt * 100 + R)
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Gradient Descent / ALS
            # We use a simple gradient step with numerical gradient or analytical?
            # Analytical gradient is better.
            # L = sum (sum_r U_a,r V_b,r W_c,r - 1)^2
            # dL/dU_a,r = 2 * (sum_r U_a,r V_b,r W_c,r - 1) * V_b,r W_c,r
            # This is expensive to compute for all a,b,c.
            # Instead, we use a simpler approach: 
            # Fix U, V, solve for W? No, it's a cubic system.
            # Let's use `scipy.optimize.minimize` on the flattened vector?
            # Size: R*(n*m + m*p + p*n) = R*(6+18+12) = 36R.
            # For R=25, size 900. This might be slow.
            
            # Alternative: Use a local search with random restarts and snapping.
            # Just do a few steps of gradient descent using NumPy vectorization.
            
            # Reshape for easier calculation?
            # Let's just do 50 steps of gradient descent with step size 0.1
            # and check error.
            
            current_err = calc_error(U, V, W)
            for step in range(50):
                if time.time() > deadline - 5:
                    break
                # Calculate gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i, j, k in indices:
                    idx_u = i*m+j
                    idx_v = j*p+k
                    idx_w = k*n+i
                    val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                    err_term = 2.0 * (val - 1.0)
                    # dU[idx_u, r] += err_term * V[idx_v, r] * W[idx_w, r]
                    dU[idx_u] += err_term * V[idx_v] * W[idx_w]
                    dV[idx_v] += err_term * U[idx_u] * W[idx_w]
                    dW[idx_w] += err_term * U[idx_u] * V[idx_v]
                
                # Update
                lr = 0.05
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                current_err = calc_error(U, V, W)
                if current_err < 1e-6:
                    break
            
            # Snap to nearest 0.5
            U_snap = np.round(U * 2.0) / 2.0
            V_snap = np.round(V * 2.0) / 2.0
            W_snap = np.round(W * 2.0) / 2.0
            
            # Check exactness
            if calc_error(U_snap, V_snap, W_snap) < 1e-9:
                best_rank = R
                best_U = U_snap
                best_V = V_snap
                best_W = W_snap
                break # Found valid for this R, try lower R? No, R is increasing. 
                       # Wait, I want the MINIMAL rank. So I should break and return.
                # But maybe a lower R exists? I am iterating R from 20 upwards.
                # So the first valid R is the minimal one found in this range.
                # I should break out of all loops.
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # If no better rank found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
