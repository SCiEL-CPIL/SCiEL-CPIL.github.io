"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def is_exact(U, V, W, n, m, p):
        """Check if the given factors exactly decompose the matrix multiplication tensor."""
        r = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def round_to_half_int(arr, tol=1e-6):
        """Round array to nearest half-integer (multiple of 0.5)"""
        rounded = np.round(arr * 2) / 2
        if np.all(np.abs(arr - rounded) < tol):
            return rounded
        return None
    
    def try_gradient_optimization(rank, max_iter=500, num_restarts=3, timeout=100):
        """
        Use gradient-based optimization to find a rank-r decomposition.
        We parameterize U, V, W as continuous variables and minimize the
        reconstruction error subject to exact constraints.
        """
        start_time = time.time()
        
        # Precompute indices for efficiency
        idx_list = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        def objective_and_grad(x):
            """
            x contains concatenated U, V, W flattened.
            Returns (loss, gradient) where loss = sum of squared errors.
            """
            if time.time() - start_time > timeout:
                return None, None
            
            U_flat = x[:n*m*rank].reshape(n*m, rank)
            V_flat = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Compute loss and gradient
            loss = 0.0
            grad_U = np.zeros_like(U_flat)
            grad_V = np.zeros_like(V_flat)
            grad_W = np.zeros_like(W_flat)
            
            for (i, j, k) in idx_list:
                # Current value
                val = 0.0
                for t in range(rank):
                    val += U_flat[i*m+j, t] * V_flat[j*p+k, t] * W_flat[k*n+i, t]
                
                error = val - 1.0
                loss += error * error
                
                # Gradients
                for t in range(rank):
                    u_val = U_flat[i*m+j, t]
                    v_val = V_flat[j*p+k, t]
                    w_val = W_flat[k*n+i, t]
                    
                    # d/dU[i*m+j, t] = 2*error * V[j*p+k, t] * W[k*n+i, t]
                    grad_U[i*m+j, t] += 2 * error * v_val * w_val
                    # d/dV[j*p+k, t] = 2*error * U[i*m+j, t] * W[k*n+i, t]
                    grad_V[j*p+k, t] += 2 * error * u_val * w_val
                    # d/dW[k*n+i, t] = 2*error * U[i*m+j, t] * V[j*p+k, t]
                    grad_W[k*n+i, t] += 2 * error * u_val * v_val
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grad
        
        best_result = None
        
        for restart in range(num_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Random initialization
            np.random.seed(42 + restart)
            x0 = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.5
            
            # Try to optimize
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'maxfun': max_iter * 10}
            )
            
            if time.time() - start_time > timeout:
                break
            
            # Extract U, V, W from result
            U_flat = result.x[:n*m*rank].reshape(n*m, rank)
            V_flat = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Check if exact
            if is_exact(U_flat, V_flat, W_flat, n, m, p):
                # Try to round to half-integers
                U_rounded = round_to_half_int(U_flat)
                V_rounded = round_to_half_int(V_flat)
                W_rounded = round_to_half_int(W_flat)
                
                if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                    if is_exact(U_rounded, V_rounded, W_rounded, n, m, p):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                
                # Return as-is if exact but not half-integer
                return {
                    "U": U_flat.tolist(),
                    "V": V_flat.tolist(),
                    "W": W_flat.tolist()
                }
            
            # Track best if not exact
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 30
    # But we can try smaller ranks
    
    best_solution = None
    
    # Try a range of ranks
    for rank in range(30, 61, 2):  # Try even ranks from 30 to 60
        if best_solution is not None and rank >= len(best_solution["U"][0]):
            break
        
        result = try_gradient_optimization(rank, max_iter=200, num_restarts=2, timeout=60)
        
        if result is not None:
            r = len(result["U"][0])
            if best_solution is None or r < len(best_solution["U"][0]):
                best_solution = result
                print(f"Found rank {r} solution")
    
    if best_solution is not None:
        return best_solution
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
