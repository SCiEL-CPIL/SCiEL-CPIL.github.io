"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target ranks to try, starting from known lower bounds
    # R(M_{2,3,6}) is known to be <= 24
    target_ranks = [24, 26, 28, 30, 32, 34, 36]
    
    # Fallback to schoolbook if nothing found
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # The tensor T[i, j, k] = 1
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Residual function for optimization
    # Variables: flattened U, V, W
    def residuals(params):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        u_flat = params[:size_u]
        v_flat = params[size_u:size_u + size_v]
        w_flat = params[size_u + size_v:]
        
        U = u_flat.reshape(n * m, rank)
        V = v_flat.reshape(m * p, rank)
        W = w_flat.reshape(p * n, rank)
        
        # Calculate product
        # C[i, k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can vectorize this.
        # Let's compute the tensor product contribution for each c
        # Or simply iterate over i, j, k
        
        res = np.zeros((n * m * p,))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    for rank in target_ranks:
        if time.time() > deadline:
            break
            
        # Initialize with random values
        np.random.seed(42) # Fixed seed for reproducibility in attempts
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_vars = size_u + size_v + size_w
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            x0 = np.random.randn(total_vars) * 0.1
            # Scale up a bit to avoid local minima at 0
            x0 *= 2.0
            
            try:
                # Use least_squares to minimize sum of squared residuals
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
                
                if result.success or result.cost < 1e-10:
                    # Extract solution
                    U = result.x[:size_u].reshape(n * m, rank)
                    V = result.x[size_u:size_u + size_v].reshape(m * p, rank)
                    W = result.x[size_u + size_v:].reshape(p * n, rank)
                    
                    # Round to nearest half-integer
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exactness
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_r[i * m + j, :] * V_r[j * p + k, :] * W_r[k * n + i, :])
                                if not np.isclose(val, 1.0, atol=1e-9):
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                    
                    if valid:
                        # Convert to list of lists for the required interface
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
            except Exception:
                continue
                
    # Fallback
    return schoolbook()
# EVOLVE-BLOCK-END
