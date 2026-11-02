import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor is all ones of shape (n, m, p)
    # We need U (n*m, R), V (m*p, R), W (p*n, R)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    # Try ranks from 20 to 36
    # We know 36 works.
    # We want to find the minimal R.
    
    deadline = time.time() + 110 # 10s margin
    
    # Helper to flatten and unflatten
    def flatten_U(U):
        return U.flatten()
    
    def unflatten_U(x):
        return x.reshape(n*m, -1)
    
    def unflatten_V(x):
        return x.reshape(m*p, -1)
    
    def unflatten_W(x):
        return x.reshape(p*n, -1)
    
    def compute_residuals(params, R):
        U = unflatten_U(params[:n*m*R]).T # R x n*m
        V = unflatten_V(params[n*m*R : n*m*R + m*p*R]).T # R x m*p
        W = unflatten_W(params[n*m*R + m*p*R :]).T # R x p*n
        
        # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We can do this efficiently with einsum or loops
        # res[i,j,k] = sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
        
        # Since n,m,p are small, a loop is fine
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vectorized dot product
                    u_vec = U[:, i*m+j]
                    v_vec = V[:, j*p+k]
                    w_vec = W[:, k*n+i]
                    res[i,j,k] = np.sum(u_vec * v_vec * w_vec)
        
        return (res - 1.0).flatten()
    
    def run_search(R, num_restarts=5, max_iter=50):
        # Initialize with random small values
        # Or use schoolbook if R >= 36
        if R >= 36:
             # Schoolbook init
             U = np.zeros((n*m, R))
             V = np.zeros((m*p, R))
             W = np.zeros((p*n, R))
             col = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U[i*m+j, col] = 1
                         V[j*p+k, col] = 1
                         W[k*n+i, col] = 1
                         col += 1
             params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
             # We can try to reduce rank by zeroing out some columns?
             # But least_squares will move them.
             # Better to start with random init for lower R.
        
        best_res = np.inf
        best_params = None
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            params = np.random.randn(n*m*R + m*p*R + p*n*R) * 0.1
            
            # Run least squares
            try:
                res = least_squares(compute_residuals, params, args=(R,), max_nfev=1000, method='lm')
                if res.cost < best_res:
                    best_res = res.cost
                    best_params = res.x
            except:
                pass
                
        if best_params is None:
            return None
            
        # Check if solution is good enough
        if best_res < 1e-6:
            # Unflatten
            U = unflatten_U(best_params[:n*m*R])
            V = unflatten_V(best_params[n*m*R : n*m*R + m*p*R])
            W = unflatten_W(best_params[n*m*R + m*p*R :])
            
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify exactness
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
                
            if valid:
                # Convert to lists
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                
        return None

    # Try ranks from low to high
    for R in range(20, 37):
        if time.time() > deadline:
            break
        result = run_search(R, num_restarts=10, max_iter=100)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
