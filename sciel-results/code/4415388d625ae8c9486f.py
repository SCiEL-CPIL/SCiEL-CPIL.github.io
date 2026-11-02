"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _build_linear_system(n, m, p, rank):
    """
    Build the linear system for tensor decomposition.
    
    We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    This is nonlinear in U,V,W. But we can fix W and solve linear for U,V,
    or use alternating linearization.
    
    Better approach: Use the fact that for fixed V and W, U is determined by
    solving a linear system. We'll use a more direct approach.
    
    For each rank r, we have variables U[:,r], V[:,r], W[:,r].
    The constraint is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    This is a system of n*m*p equations in (n*m + m*p + p*n)*rank unknowns.
    
    We can solve this using a nonlinear solver, or we can use the structure
    to create a more efficient approach.
    
    Let's use a different strategy: fix the structure and solve for the values.
    """
    pass

def _solve_via_alternating_linearization(n, m, p, rank, max_iter=500, tol=1e-10, seed=None):
    """
    Solve using alternating linearization:
    - Fix V and W, solve linear for U
    - Fix U and W, solve linear for V  
    - Fix U and V, solve linear for W
    """
    if time.time() > _DEADLINE - 5.0:
        return None
    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_obj = float('inf')
    best_state = None
    best_residual = None
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Update U: for each (i,j), solve A @ U[i*m+j,:] = 1
        # where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    # Solve A @ x = ones(p)
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve B @ V[j*p+k,:] = 1
        # where B[i, r] = U[i*m+j, r] * W[k*n+i, r]
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve C @ W[k*n+i,:] = 1
        # where C[j, r] = U[i*m+j, r] * V[j*p+k, r]
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Compute residual
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    residual += (s - 1.0) ** 2
        
        if residual < best_obj:
            best_obj = residual
            best_state = (U.copy(), V.copy(), W.copy())
            best_residual = residual
        
        if residual < tol:
            break
    
    if best_obj < tol:
        U, V, W = best_state
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list
    
    return None

def _try_sparse_linear_system(n, m, p, rank, seed=None):
    """
    Try to solve using a more structured approach.
    
    Key insight: The matrix multiplication tensor has a specific structure.
    For 3x4x6, we can try to find a decomposition where each rank-1 term
    corresponds to a specific pattern.
    
    We'll use a nonlinear optimization approach with better initialization.
    """
    if time.time() > _DEADLINE - 5.0:
        return None
    
    if seed is not None:
        np.random.seed(seed)
    
    # Total number of variables
    num_u = n * m
    num_v = m * p
    num_w = p * n
    total_vars = (num_u + num_v + num_w) * rank
    
    def flatten_params(x):
        U = x[:num_u*rank].reshape(num_u, rank)
        V = x[num_u*rank:(num_u+num_v)*rank].reshape(num_v, rank)
        W = x[(num_u+num_v)*rank:].reshape(num_w, rank)
        return U, V, W
    
    def objective(x):
        U, V, W = flatten_params(x)
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    obj += (s - 1.0) ** 2
        return obj
    
    # Initialize with random values
    x0 = np.random.randn(total_vars) * 0.1
    
    # Try to solve
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
    
    if result.success or result.fun < 1e-10:
        U, V, W = flatten_params(result.x)
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
            return U_list, V_list, W_list
    
    return None

def _try_rank_combined(n, m, p, rank, num_seeds=5, tol=1e-8):
    """Try multiple approaches to find a rank-rank decomposition."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    # Try alternating linearization with multiple seeds
    for seed in range(num_seeds):
        if time.time() > _DEADLINE - 5.0:
            break
        
        result = _solve_via_alternating_linearization(n, m, p, rank, max_iter=300, tol=tol, seed=seed)
        if result is not None:
            return result
    
    # Try nonlinear optimization with a few seeds
    for seed in range(2):
        if time.time() > _DEADLINE - 5.0:
            break
        
        result = _try_sparse_linear_system(n, m, p, rank, seed=seed)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback
    
    # Lower bound: max(n*m, m*p, n*p) = max(12, 24, 18) = 24
    # Try from 24 up to 45
    
    for rank in range(24, 46):
        if time.time() > _DEADLINE - 10.0:
            break
        
        result = _try_rank_combined(n, m, p, rank, num_seeds=8, tol=1e-8)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best_result = (U, V, W)
                best_rank = rank
                break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
