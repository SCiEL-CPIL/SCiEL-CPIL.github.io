"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # safety margin
    
    # Build the matrix multiplication tensor
    # T[I, J, K] = 1 where I=i*m+j, J=j'*p+k, K=k'*n+i'
    # with j=j', i=i', k=k'
    # So for each (i, j, k): T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    def compute_residual(U, V, W):
        """Compute R = sum_c U[:,c] V[:,c] W[:,c] - T"""
        R = np.einsum('ic,jc,kc->ijk', U, V, W)
        R -= T
        return R
    
    def objective_and_grad(params, r):
        """Compute objective and gradient for BFGS optimization."""
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U = params[:size_U].reshape(n*m, r)
        V = params[size_U:size_U+size_V].reshape(m*p, r)
        W = params[size_U+size_V:].reshape(p*n, r)
        
        R = compute_residual(U, V, W)
        loss = np.sum(R**2)
        
        # Gradients
        grad_U = 2.0 * np.einsum('ijk,jc,kc->ic', R, V, W)
        grad_V = 2.0 * np.einsum('ijk,ic,kc->jc', R, U, W)
        grad_W = 2.0 * np.einsum('ijk,ic,jc->kc', R, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad
    
    def check_half_integer_solution(U, V, W, tol=1e-6):
        """Check if solution is close to half-integers and exact."""
        # Round to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check closeness to half-integers
        if np.max(np.abs(U - U_r)) > tol:
            return False, None, None, None
        if np.max(np.abs(V - V_r)) > tol:
            return False, None, None, None
        if np.max(np.abs(W - W_r)) > tol:
            return False, None, None, None
        
        # Verify exactness with rounded values
        R = compute_residual(U_r, V_r, W_r)
        if np.max(np.abs(R)) < 1e-9:
            return True, U_r, V_r, W_r
        return False, None, None, None
    
    # Try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But tensor rank is typically higher. Start from 20 to be safe.
    # Known: 3x3x3 has rank 23. For 3x3x5, expect rank around 30-40.
    
    best_result = None
    
    from scipy.optimize import minimize
    
    # Try different ranks, starting from a reasonable estimate
    # For 3x3x5, let's try from 28 up to 45
    for r in range(28, 46):
        if time.time() > deadline - 5:
            break
        
        found = False
        num_restarts = 3
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            np.random.seed(42 + restart * 100 + r)
            scale = 0.5
            params = np.random.randn(r * (n*m + m*p + p*n)) * scale
            
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, r)[0],
                    params,
                    jac=lambda x: objective_and_grad(x, r)[1],
                    method='BFGS',
                    options={'gtol': 1e-8, 'maxiter': 100, 'disp': False}
                )
                
                if res.fun < 1e-6:
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    is_exact, U_r, V_r, W_r = check_half_integer_solution(U, V, W)
                    if is_exact:
                        # Convert to lists
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        
            except Exception:
                pass
    
    # Fallback: schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, column] = 1
                V[j*p + k, column] = 1
                W[k*n + i, column] = 1
                column += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
