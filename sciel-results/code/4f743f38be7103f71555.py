import numpy as np
import time
from scipy.optimize import minimize

# ... existing code ...

def try_find_rank_43(n, m, p, time_limit=100):
    """
    Attempts to find a rank 43 decomposition using random initialization and optimization.
    Returns None if not found within time limit.
    """
    start_time = time.time()
    R = 43
    # Initialize with random small values
    # Or use the rank 44 solution and drop the last column?
    # Dropping a column from a valid decomposition won't be valid.
    # Random init is better for escape from local minima, but harder.
    # Let's try random init with scaling.
    
    U = np.random.randn(n * m, R)
    V = np.random.randn(m * p, R)
    W = np.random.randn(p * n, R)
    
    # The tensor T is 1.0 at T[i, j, k] if k == j? No.
    # T[i, j, k] = 1 if the multiplication A[i, j] * B[j, k] contributes to C[k, i]?
    # Wait, the convention is W[k*n+i, r].
    # The tensor T[i, j, k] is 1 for all i, j, k?
    # No, the matrix multiplication tensor T has T[i, j, k] = 1 if the term A[i, j] B[j, k] is present.
    # In standard matrix multiplication, ALL terms are present.
    # So T[i, j, k] = 1 for all i, j, k.
    # Wait, is that right?
    # C[k, i] = sum_j A[i, j] B[j, k].
    # So the tensor is T[i, j, k] = 1.
    # Yes, the "multiplication tensor" is the all-ones tensor of shape (n, m, p).
    
    T = np.ones((n, m, p))
    
    def objective(params):
        # Unpack params into U, V, W
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n * m, R)
        V = V_flat.reshape(m * p, R)
        W = W_flat.reshape(p * n, R)
        
        # Compute reconstructed tensor
        # T_recon[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is expensive.
        # We can use einsum.
        # T_recon = np.einsum('ir, jr, kr -> ijk', U, V, W) ??
        # Indices:
        # U: (n*m, R) -> index i*m+j
        # V: (m*p, R) -> index j*p+k
        # W: (p*n, R) -> index k*n+i
        
        # Let's reshape U, V, W to 3D
        # U3[i, j, r] = U[i*m+j, r]
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # T_recon[i, j, k] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
        # Note the index mapping for W: W[k*n+i, r] -> W3[k, i, r]
        
        # einsum:
        # U3[i, j, r]
        # V3[j, k, r]
        # W3[k, i, r]
        # Result: T[i, j, k]
        
        T_recon = np.einsum('ijr,jkr,ki r->ijk'.replace(' ', ''), U3, V3, W3)
        
        diff = T_recon - T
        return np.sum(diff * diff)
    
    # Initial guess
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Use Nelder-Mead or BFGS?
    # BFGS is better for smooth problems.
    # But it's high dimensional.
    # n*m*R = 12*43 = 516
    # m*p*R = 16*43 = 688
    # p*n*R = 12*43 = 516
    # Total ~ 1700 variables.
    # BFGS might be slow.
    
    # Let's try a few random restarts.
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Scale down to avoid overflow
        x0 = x0 * 0.1
        
        res = minimize(objective, x0, method='BFGS', options={'maxiter': 100, 'gtol': 1e-5})
        
        if res.fun < 1e-9:
            # Found a solution!
            U_sol = res.x[:n*m*R].reshape(n*m, R)
            V_sol = res.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
            W_sol = res.x[n*m*R+m*p*R:].reshape(p*n, R)
            return U_sol, V_sol, W_sol
            
    return None

def construct():
    # ... existing code ...
    # Try to find rank 43
    best_rank = 44
    best_U, best_V, best_W = U, V, W # From existing logic
    
    # Use remaining time to try rank 43
    time_elapsed = time.time() - start_time
    if time_elapsed < 120:
        res = try_find_rank_43(n, m, p, time_limit=110 - time_elapsed)
        if res is not None:
            U_sol, V_sol, W_sol = res
            # Verify
            if _verify(U_sol, V_sol, W_sol, n, m, p):
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                
    return {"U": U, "V": V, "W": W}
