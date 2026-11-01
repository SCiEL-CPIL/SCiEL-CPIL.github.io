import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Try to find a lower rank decomposition
    # Target rank is 54 based on inspiration scores
    target_rank = 54
    
    # Prepare indices for non-zero and zero entries
    # Non-zero: (i, j, k) -> u=i*m+j, v=j*p+k, w=k*n+i
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                nonzero_indices.append((u, v, w))
    
    # Zero entries: all other (u, v, w)
    # Total entries: n*m * m*p * p*n
    zero_indices = []
    # This loop is 10 * 30 * 12 = 3600 iterations, which is fine
    nonzero_set = set(nonzero_indices)
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) not in nonzero_set:
                    zero_indices.append((u, v, w))
                    
    # Total equations
    num_eq = len(nonzero_indices) + len(zero_indices)
    
    # Variables: U, V, W flattened
    # U: n*m x R
    # V: m*p x R
    # W: p*n x R
    num_vars = (n * m + m * p + p * n) * target_rank
    
    def loss_func(params):
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = params[n*m*target_rank + m*p*target_rank :]
        
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        
        errors = np.zeros(num_eq)
        idx = 0
        for u, v, w in nonzero_indices:
            # Sum over r
            s = 0.0
            for r in range(target_rank):
                s += U[u, r] * V[v, r] * W[w, r]
            errors[idx] = s - 1.0
            idx += 1
            
        for u, v, w in zero_indices:
            s = 0.0
            for r in range(target_rank):
                s += U[u, r] * V[v, r] * W[w, r]
            errors[idx] = s
            idx += 1
            
        return errors

    # Initialize parameters
    np.random.seed(42)
    x0 = np.random.randn(num_vars) * 0.1
    
    # Solve using least_squares
    try:
        result = least_squares(loss_func, x0, method='trf', max_nfev=10000)
        if result.success and np.linalg.norm(result.fun) < 1e-5:
            # Extract solution
            U_flat = result.x[:n*m*target_rank]
            V_flat = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = result.x[n*m*target_rank + m*p*target_rank :]
            
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            max_err = 0
            for u, v, w in nonzero_indices:
                s = 0.0
                for r in range(target_rank):
                    s += U_rounded[u, r] * V_rounded[v, r] * W_rounded[w, r]
                max_err = max(max_err, abs(s - 1.0))
                
            for u, v, w in zero_indices:
                s = 0.0
                for r in range(target_rank):
                    s += U_rounded[u, r] * V_rounded[v, r] * W_rounded[w, r]
                max_err = max(max_err, abs(s))
                
            if max_err < 1e-9:
                # Convert to lists
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
                
    except Exception as e:
        pass
        
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
