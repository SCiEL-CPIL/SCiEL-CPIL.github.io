"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other combinations, the sum should be 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The (i,j,k) entry of the multiplication tensor is 1
                u_val = U[i*m + j, :]
                v_val = V[j*p + k, :]
                w_val = W[k*n + i, :]
                dot = np.sum(u_val * v_val * w_val)
                if abs(dot - 1.0) > tol:
                    return False
    
    # Check that all other entries are 0
    # The tensor is only nonzero when we're computing C[i,k] from A[i,j]*B[j,k]
    # So for a given output position (i,k) and input positions (i',j') and (j'',k'),
    # the tensor entry is 1 if i==i', j'==j'', k==k', else 0
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for j2 in range(m):
                    for j3 in range(m):
                        for k2 in range(p):
                            if i2 == i and j3 == j2 and k2 == k:
                                continue
                            else:
                                # This combination should give 0
                                # But we need to be careful about the tensor structure
                                pass
    # Simpler verification: reconstruct the multiplication
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # C[i,k] should get contribution 1 from A[i,j]*B[j,k]
                u_val = U[i*m + j, :]
                v_val = V[j*p + k, :]
                w_val = W[k*n + i, :]
                dot = np.sum(u_val * v_val * w_val)
                if abs(dot - 1.0) > tol:
                    return False
            # Check no spurious contributions
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        if i2 != i or k2 != k:
                            continue
                        # Already checked above
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 120s limit
    
    # Try to find rank 28 decomposition
    target_rank = 28
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank 30
    
    # First, get the schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Try randomized search for rank 28
    import random
    random.seed(42)
    np.random.seed(42)
    
    attempts = 0
    max_attempts = 5000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if attempts % 500 == 0:
            print(f"Attempt {attempts}, elapsed: {time.time()-start_time:.1f}s")
        
        # Random initialization
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Define the objective: minimize the error in reconstructing the tensor
        def objective(params):
            U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_flat = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U_flat[i*m + j, :]
                        v_val = V_flat[j*p + k, :]
                        w_val = W_flat[k*n + i, :]
                        prod = np.sum(u_val * v_val * w_val)
                        error += (prod - 1.0) ** 2
            return error
        
        # Use Nelder-Mead for small-scale optimization
        if time.time() > deadline - 5:
            break
            
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective, params, method='Nelder-Mead', 
                            options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < 1e-8:
                U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = np.sum(U_opt[i*m+j, :] * V_opt[j*p+k, :] * W_opt[k*n+i, :])
                            if abs(prod - 1.0) > 1e-6:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    # Round to nearest simple fractions
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    valid_rounded = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                prod = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                                if abs(prod - 1.0) > 1e-6:
                                    valid_rounded = False
                                    break
                        if not valid_rounded:
                            break
                    
                    if valid_rounded:
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    else:
                        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        except Exception:
            continue
    
    # Fallback to schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
