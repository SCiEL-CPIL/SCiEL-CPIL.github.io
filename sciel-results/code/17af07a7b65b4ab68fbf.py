"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
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
        r = U.shape[1]
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
    
    def try_gradient_descent(rank, num_restarts=5, max_iter=2000):
        """Use gradient-based optimization to find a rank-decomposition."""
        np.random.seed(42)
        
        # Total number of variables: (n*m + m*p + p*n) * rank
        num_vars = (n * m + m * p + p * n) * rank
        
        def objective_and_grad(x):
            """Objective: sum of squared errors for all (i,j,k) constraints."""
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for t in range(rank):
                            s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                        error += (s - 1.0) ** 2
            
            return error
        
        def gradient(x):
            """Compute gradient of the objective."""
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for t in range(rank):
                            s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                        err = 2.0 * (s - 1.0)
                        for t in range(rank):
                            grad_U[i*m+j, t] += err * V[j*p+k, t] * W[k*n+i, t]
                            grad_V[j*p+k, t] += err * U[i*m+j, t] * W[k*n+i, t]
                            grad_W[k*n+i, t] += err * U[i*m+j, t] * V[j*p+k, t]
            
            grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return grad_flat
        
        best_result = None
        best_error = float('inf')
        
        for restart in range(num_restarts):
            # Random initialization
            x0 = np.random.randn(num_vars) * 0.5
            
            result = minimize(
                objective_and_grad,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_error:
                best_error = result.fun
                best_result = result
        
        if best_result is None:
            return None
        
        if best_result.fun > 1e-6:
            return None
        
        # Extract solution
        U = best_result.x[:n*m*rank].reshape(n*m, rank)
        V = best_result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = best_result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Try to round to integers or half-integers
        def round_to_half(x):
            return np.round(2 * x) / 2.0
        
        U_rounded = round_to_half(U)
        V_rounded = round_to_half(V)
        W_rounded = round_to_half(W)
        
        if is_exact(U_rounded, V_rounded, W_rounded, n, m, p):
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
        
        # Try rounding to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        
        if is_exact(U_int, V_int, W_int, n, m, p):
            return {
                "U": U_int.tolist(),
                "V": V_int.tolist(),
                "W": W_int.tolist()
            }
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 30
    lower_bound = max(n*m, m*p, p*n)
    schoolbook_rank = n * m * p
    
    # Try a few candidate ranks
    candidate_ranks = []
    for r in range(lower_bound, min(lower_bound + 15, schoolbook_rank + 1)):
        candidate_ranks.append(r)
    
    # Also try some specific values
    for r in [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58]:
        if r not in candidate_ranks and r >= lower_bound and r <= schoolbook_rank:
            candidate_ranks.append(r)
    
    candidate_ranks = sorted(set(candidate_ranks))
    
    for rank in candidate_ranks:
        result = try_gradient_descent(rank, num_restarts=3, max_iter=1000)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
