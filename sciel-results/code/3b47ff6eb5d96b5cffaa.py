import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Trivial decomposition
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((m * p, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((p * n, trivial_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
                
    # Store non-zero indices of the target tensor
    indices_I = []
    indices_J = []
    indices_K = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices_I.append(i * m + j)
                indices_J.append(j * p + k)
                indices_K.append(k * n + i)
                
    indices_I = np.array(indices_I)
    indices_J = np.array(indices_J)
    indices_K = np.array(indices_K)
    target_vals = np.ones(len(indices_I))
    
    def compute_error(U, V, W):
        U_I = U[indices_I]
        V_J = V[indices_J]
        W_K = W[indices_K]
        vals = (U_I * V_J * W_K).sum(axis=1)
        return np.sum((vals - target_vals) ** 2)

    def verify_decomposition(U, V, W, tol=1e-9):
        U_I = U[indices_I]
        V_J = V[indices_J]
        W_K = W[indices_K]
        vals = (U_I * V_J * W_K).sum(axis=1)
        return np.all(np.abs(vals - target_vals) < tol)

    def check_half_integer(U, V, W, tol=1e-9):
        for mat in [U, V, W]:
            if not np.all(np.abs(mat * 2 - np.round(mat * 2)) < tol):
                return False
        return True

    def snap_to_half_integer(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap

    def try_rank_algebraic(R, num_trials=100):
        """Try to find a rank-R decomposition using algebraic structure"""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            np.random.seed(12345 + R * 100 + trial)
            
            # Initialize with random values scaled appropriately
            scale = 0.5
            U = np.random.randn(n * m, R) * scale
            V = np.random.randn(m * p, R) * scale
            W = np.random.randn(p * n, R) * scale
            
            # Use a more sophisticated optimization: alternating minimization
            # Fix two matrices, solve for the third
            lr = 0.1
            for step in range(500):
                if time.time() > deadline:
                    return None
                
                # Compute current error
                err = compute_error(U, V, W)
                if err < 1e-12:
                    break
                
                # Alternating least squares approach
                # For each r, we want to minimize the residual
                # This is a tensor decomposition problem
                
                # Update U: for each row i, minimize ||target_i - U[i,:] @ (V*W)_i||^2
                # (V*W)_i is the element-wise product of V and W rows corresponding to i
                
                # Compute effective target for U
                # T[I_s, r] = sum over other indices
                # This is complex, so let's use gradient-based approach with adaptive step
                
                U_I = U[indices_I]
                V_J = V[indices_J]
                W_K = W[indices_K]
                vals = (U_I * V_J * W_K).sum(axis=1)
                diff = vals - target_vals
                
                # Gradient for U
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for s in range(len(indices_I)):
                    I = indices_I[s]
                    J = indices_J[s]
                    K = indices_K[s]
                    d = diff[s]
                    
                    dU[I, :] += d * V[J, :] * W[K, :]
                    dV[J, :] += d * U[I, :] * W[K, :]
                    dW[K, :] += d * U[I, :] * V[J, :]
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 1e-10:
                    step_size = lr / grad_norm
                    U -= step_size * dU
                    V -= step_size * dV
                    W -= step_size * dW
                
                # Decay learning rate
                if step % 50 == 0:
                    lr *= 0.95
                
                # Periodic snapping to help with half-integer constraint
                if step % 100 == 0 and step > 0:
                    U, V, W = snap_to_half_integer(U, V, W)
                    err = compute_error(U, V, W)
                    if err < 1e-9:
                        if verify_decomposition(U, V, W) and check_half_integer(U, V, W):
                            return (U, V, W)
            
            # Final check
            err = compute_error(U, V, W)
            if err < 1e-9:
                U, V, W = snap_to_half_integer(U, V, W)
                if verify_decomposition(U, V, W) and check_half_integer(U, V, W):
                    return (U, V, W)
        
        return None

    def try_rank_structured(R):
        """Try structured decompositions based on algebraic identities"""
        if time.time() > deadline - 5:
            return None
        
        # For 2x3x4, we can try to exploit the structure
        # The key insight: if we can find a linear combination of terms that cancels out
        # certain contributions, we can reduce the rank
        
        # One approach: start from rank 24 and try to merge terms
        # This is equivalent to finding a rank-R decomposition directly
        
        # Let's try a different initialization: use the trivial decomposition
        # and randomly perturb, then optimize
        for trial in range(50):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(99999 + R * 7 + trial)
            
            # Start close to trivial but with reduced rank
            # This is tricky - let's just use random init with different scale
            scale = np.random.choice([0.1, 0.25, 0.5, 0.75])
            U = np.random.randn(n * m, R) * scale
            V = np.random.randn(m * p, R) * scale
            W = np.random.randn(p * n, R) * scale
            
            # Use L-BFGS-like approach with multiple restarts
            best_err = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(300):
                if time.time() > deadline - 5:
                    break
                
                U_I = U[indices_I]
                V_J = V[indices_J]
                W_K = W[indices_K]
                vals = (U_I * V_J * W_K).sum(axis=1)
                diff = vals - target_vals
                err = np.sum(diff**2)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-12:
                    break
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for s in range(len(indices_I)):
                    I = indices_I[s]
                    J = indices_J[s]
                    K = indices_K[s]
                    d = diff[s]
                    
                    dU[I, :] += d * V[J, :] * W[K, :]
                    dV[J, :] += d * U[I, :] * W[K, :]
                    dW[K, :] += d * U[I, :] * V[J, :]
                
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 1e-10:
                    step_size = 0.05 / grad_norm
                    U -= step_size * dU
                    V -= step_size * dV
                    W -= step_size * dW
            
            # Check best solution
            if best_err < 1e-9:
                U, V, W = snap_to_half_integer(best_U, best_V, best_W)
                if verify_decomposition(U, V, W) and check_half_integer(U, V, W):
                    return (U, V, W)
        
        return None

    # Try ranks from 23 down to 19
    for R in range(23, 18, -1):
        if time.time() > deadline - 15:
            break
        
        # First try structured approach
        result = try_rank_structured(R)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
        # Then try algebraic approach
        result = try_rank_algebraic(R, num_trials=50)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to trivial
    return {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
