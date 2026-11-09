"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _compute_tensor_values(U, V, W, n, m, p, rank):
    """Compute all tensor values for verification."""
    values = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                values[i, j, k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
    return values

def _verify_solution(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify if the solution is exact (within tolerance) and has integer/half-integer entries."""
    values = _compute_tensor_values(U, V, W, n, m, p, rank)
    max_err = np.max(np.abs(values - 1.0))
    
    if max_err > tol:
        return False
    
    # Check for integer/half-integer entries
    for mat in [U, V, W]:
        for val in mat.flatten():
            if abs(val - round(val)) > 1e-7:
                if abs(val * 2 - round(val * 2)) > 1e-7:
                    return False
    return True

def _try_rank_structured(n, m, p, rank, time_limit=3.0):
    """Try to find a rank-rank solution using structured random initializations."""
    start_time = time.time()
    n_params = (n*m + m*p + p*n) * rank
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        
        np.random.seed(attempt * 7919 + rank * 13)
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for r in range(rank):
            for i in range(n*m):
                if np.random.random() < 0.3:
                    U[i, r] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
            
            for j in range(m*p):
                if np.random.random() < 0.3:
                    V[j, r] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
            
            for k in range(p*n):
                if np.random.random() < 0.3:
                    W[k, r] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_x = x[:n*m*rank].reshape(n*m, rank)
            V_x = x[n*m*rank:(n*m + m*p)*rank].reshape(m*p, rank)
            W_x = x[(n*m + m*p)*rank:].reshape(p*n, rank)
            values = _compute_tensor_values(U_x, V_x, W_x, n, m, p, rank)
            return np.sum((values - 1.0)**2)
        
        def gradient(x):
            U_x = x[:n*m*rank].reshape(n*m, rank)
            V_x = x[n*m*rank:(n*m + m*p)*rank].reshape(m*p, rank)
            W_x = x[(n*m + m*p)*rank:].reshape(p*n, rank)
            
            grad_U = np.zeros_like(U_x)
            grad_V = np.zeros_like(V_x)
            grad_W = np.zeros_like(W_x)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = 2 * (np.dot(U_x[i*m+j], V_x[j*p+k] * W_x[k*n+i]) - 1.0)
                        grad_U[i*m+j] += err * V_x[j*p+k] * W_x[k*n+i]
                        grad_V[j*p+k] += err * U_x[i*m+j] * W_x[k*n+i]
                        grad_W[k*n+i] += err * U_x[i*m+j] * V_x[j*p+k]
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        try:
            result = minimize(objective, params, jac=gradient, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < 1e-8:
                U_opt = result.x[:n*m*rank].reshape(n*m, rank)
                V_opt = result.x[n*m*rank:(n*m + m*p)*rank].reshape(m*p, rank)
                W_opt = result.x[(n*m + m*p)*rank:].reshape(p*n, rank)
                
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                if _verify_solution(U_rounded, V_rounded, W_rounded, n, m, p, rank):
                    return U_rounded, V_rounded, W_rounded
                else:
                    if _verify_solution(U_opt, V_opt, W_opt, n, m, p, rank):
                        return U_opt, V_opt, W_opt
        except Exception:
            continue
    
    return None

def _try_rank_random(n, m, p, rank, time_limit=3.0):
    """Try to find a rank-rank solution using dense random initializations."""
    start_time = time.time()
    n_params = (n*m + m*p + p*n) * rank
    
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        
        np.random.seed(attempt * 104729 + rank * 31)
        params = np.random.randn(n_params) * 0.3
        
        def objective(x):
            U_x = x[:n*m*rank].reshape(n*m, rank)
            V_x = x[n*m*rank:(n*m + m*p)*rank].reshape(m*p, rank)
            W_x = x[(n*m + m*p)*rank:].reshape(p*n, rank)
            values = _compute_tensor_values(U_x, V_x, W_x, n, m, p, rank)
            return np.sum((values - 1.0)**2)
        
        def gradient(x):
            U_x = x[:n*m*rank].reshape(n*m, rank)
            V_x = x[n*m*rank:(n*m + m*p)*rank].reshape(m*p, rank)
            W_x = x[(n*m + m*p)*rank:].reshape(p*n, rank)
            
            grad_U = np.zeros_like(U_x)
            grad_V = np.zeros_like(V_x)
            grad_W = np.zeros_like(W_x)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = 2 * (np.dot(U_x[i*m+j], V_x[j*p+k] * W_x[k*n+i]) - 1.0)
                        grad_U[i*m+j] += err * V_x[j*p+k] * W_x[k*n+i]
                        grad_V[j*p+k] += err * U_x[i*m+j] * W_x[k*n+i]
                        grad_W[k*n+i] += err * U_x[i*m+j] * V_x[j*p+k]
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        try:
            result = minimize(objective, params, jac=gradient, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < 1e-8:
                U_opt = result.x[:n*m*rank].reshape(n*m, rank)
                V_opt = result.x[n*m*rank:(n*m + m*p)*rank].reshape(m*p, rank)
                W_opt = result.x[(n*m + m*p)*rank:].reshape(p*n, rank)
                
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                if _verify_solution(U_rounded, V_rounded, W_rounded, n, m, p, rank):
                    return U_rounded, V_rounded, W_rounded
                else:
                    if _verify_solution(U_opt, V_opt, W_opt, n, m, p, rank):
                        return U_opt, V_opt, W_opt
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    max_rank = n * m * p
    
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_result = (U, V, W)
    best_rank = max_rank
    
    for rank in range(max_rank - 1, 19, -1):
        if time.time() > deadline - 15:
            break
        
        time_left = deadline - time.time() - 15
        if time_left < 2:
            break
        
        time_limit = min(4.0, time_left * 0.5)
        
        result = _try_rank_structured(n, m, p, rank, time_limit=time_limit)
        
        if result is None and time.time() < deadline - 10:
            result = _try_rank_random(n, m, p, rank, time_limit=min(3.0, deadline - time.time() - 10))
        
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = rank
            break
    
    U, V, W = best_result
    
    def to_clean_list(mat):
        result = []
        for row in mat:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-7:
                    clean_row.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-7:
                    clean_row.append(float(int(round(val * 2))) / 2.0)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result
    
    return {"U": to_clean_list(U), "V": to_clean_list(V), "W": to_clean_list(W)}
# EVOLVE-BLOCK-END
