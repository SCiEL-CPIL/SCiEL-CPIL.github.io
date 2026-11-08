"""Program for 3x3x6 matrix multiplication rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _build_tensor_constraints(n, m, p):
    """Return list of (U_idx, V_idx, W_idx) for each tensor entry that must equal 1."""
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
    return constraints

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact. Returns max error."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close."""
    rounded = np.round(x * 2) / 2.0
    if np.max(np.abs(x - rounded)) < tol:
        return rounded
    return x

def _objective(x, n, m, p, rank):
    U_flat = x[:n*m*rank]
    V_flat = x[n*m*rank : n*m*rank + m*p*rank]
    W_flat = x[n*m*rank + m*p*rank :]
    
    U_mat = U_flat.reshape(n * m, rank)
    V_mat = V_flat.reshape(m * p, rank)
    W_mat = W_flat.reshape(p * n, rank)
    
    err_sum = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                err_sum += (val - 1.0) ** 2
    return err_sum

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions
    # Focus on ranks that are plausible (e.g., 46, 45, 44)
    # We use a limited number of attempts per rank to stay within time
    rank_targets = [46, 45, 44, 43, 42, 41, 40]
    
    for target_rank in rank_targets:
        if time.time() - start_time > deadline - 5:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() - start_time > deadline - 5:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialization: Random, but scaled
            U_init = np.random.randn(n * m, target_rank) * 0.5
            V_init = np.random.randn(m * p, target_rank) * 0.5
            W_init = np.random.randn(p * n, target_rank) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(_objective, x0, args=(n, m, p, target_rank), 
                                  method='L-BFGS-B', 
                                  options={'maxiter': 100, 'ftol': 1e-12})
                
                if result.fun < 1e-8:
                    U_found = result.x[:n*m*target_rank].reshape(n * m, target_rank)
                    V_found = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
                    W_found = result.x[n*m*target_rank + m*p*target_rank :].reshape(p * n, target_rank)
                    
                    # Verify
                    err = _check_decomposition(U_found, V_found, W_found, n, m, p)
                    if err < 1e-6:
                        # Try to round to half-integers
                        U_rounded = _to_half_int(U_found)
                        V_rounded = _to_half_int(V_found)
                        W_rounded = _to_half_int(W_found)
                        
                        if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p) < 1e-6:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        else:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_found, V_found, W_found
                        
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            # If we found a valid decomposition at this rank, we can try to go lower
            # but for now we break to save time for higher priority ranks or just return
            # Actually, we should continue to try lower ranks if time permits
            pass

    # Final result
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
