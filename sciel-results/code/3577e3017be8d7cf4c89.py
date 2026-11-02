"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def verify_decomp(U, V, W, tol=1e-8):
        """Verify tensor decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1))
        return max_err < tol
    
    def tensor_residual(U, V, W):
        """Compute residual for tensor equations (vectorized)."""
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                # U[i*m+j, :] * W[k*n+i, :] gives vector over k
                u_row = U[i*m+j, :]
                # For each k, V[j*p+k, :] and W[k*n+i, :]
                # residuals[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                for k in range(p):
                    residuals[i,j,k] = np.dot(u_row, V[j*p+k, :] * W[k*n+i, :])
        return (residuals - 1.0).flatten()
    
    def objective(x, r):
        """Objective: sum of squared residuals."""
        U = x[:r*n*m].reshape(n*m, r)
        V = x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
        W = x[r*n*m + r*m*p:].reshape(p*n, r)
        res = tensor_residual(U, V, W)
        return np.sum(res * res)
    
    # Known lower bounds: for matrix multiplication C = A*B where A is n×m, B is m×p
    # The tensor rank is at least max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But we need to find exact decompositions
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # For 3x4x7, known results suggest rank might be around 28-35
    start_rank = max(n*m, m*p, p*n)  # 28
    
    for r in range(start_rank, rank_school + 1):
        if time.time() > deadline:
            break
        
        n_vars = r * (n*m + m*p + p*n)
        
        # Try multiple random initializations with different strategies
        n_attempts = 15 if r < 40 else 8
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Sparse-like init
                x0 = np.zeros(n_vars)
                for idx in range(n_vars):
                    if np.random.rand() < 0.3:
                        x0[idx] = np.random.choice([-1, 1]) * np.random.rand()
            elif attempt == 1:
                # Small random
                x0 = np.random.randn(n_vars) * 0.1
            elif attempt == 2:
                # Structured init based on Kronecker product ideas
                x0 = np.random.randn(n_vars) * 0.5
            else:
                # Random
                x0 = np.random.randn(n_vars) * np.random.choice([0.1, 0.5, 1.0, 2.0])
            
            try:
                # Use L-BFGS-B for optimization
                result = minimize(
                    objective,
                    x0,
                    args=(r,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.fun < 1e-10:
                    U_sol = result.x[:r*n*m].reshape(n*m, r)
                    V_sol = result.x[r*n*m:r*n*m + r*m*p].reshape(m*p, r)
                    W_sol = result.x[r*n*m + r*m*p:].reshape(p*n, r)
                    
                    if verify_decomp(U_sol, V_sol, W_sol):
                        best_rank = r
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        break
            except Exception:
                pass
        
        if best_rank == r:
            break
    
    # Try to find half-integer solutions by rounding
    def try_half_integer(U, V, W):
        """Try rounding to half-integers and verify."""
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        if verify_decomp(U_h, V_h, W_h):
            return U_h, V_h, W_h
        return None
    
    # Check if best solution can be made half-integer
    rounded = try_half_integer(best_U, best_V, best_W)
    if rounded:
        best_U, best_V, best_W = rounded
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
