import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_shape = (n * m, rank_schoolbook)
    V_shape = (m * p, rank_schoolbook)
    W_shape = (p * n, rank_schoolbook)
    
    # Generate schoolbook solution as fallback
    U_sb = np.zeros(U_shape)
    V_sb = np.zeros(V_shape)
    W_sb = np.zeros(W_shape)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Function to verify if a set of matrices U, V, W with rank r is a valid decomposition
    def verify(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        # Also check entries are close to half-integers
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        # Check half-integer constraint
        for mat in [U, V, W]:
            for val in mat:
                if abs(val * 2 - round(val * 2)) > 1e-6:
                    return False
        return True

    # Search for lower rank
    start_time = time.time()
    time_limit = 10.0 # seconds
    
    # Try rank 23
    target_rank = rank_schoolbook - 1
    if target_rank < 1:
        target_rank = 1
        
    # We will try to find a solution for target_rank
    # Number of variables: n*m*r + m*p*r + p*n*r
    num_vars = (n*m + m*p + p*n) * target_rank
    
    def unpack_vars(x):
        # x is flat array
        U_flat = x[:n*m*target_rank]
        V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = x[n*m*target_rank + m*p*target_rank :]
        
        U = U_flat.reshape((n*m, target_rank))
        V = V_flat.reshape((m*p, target_rank))
        W = W_flat.reshape((p*n, target_rank))
        return U, V, W

    def objective(x):
        U, V, W = unpack_vars(x)
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(target_rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res += (val - 1.0)**2
        return res

    # Random restarts
    best_solution = None
    np.random.seed(42)
    
    while time.time() - start_time < time_limit:
        # Initialize with small random values
        x0 = np.random.randn(num_vars) * 0.5
        
        # Use L-BFGS-B
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.success and res.fun < 1e-8:
                U, V, W = unpack_vars(res.x)
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if verify(U_snap, V_snap, W_snap):
                    best_solution = (U_snap, V_snap, W_snap)
                    break
        except Exception:
            pass
            
    if best_solution is not None:
        U, V, W = best_solution
        # Convert to lists of lists as required
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Return schoolbook
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
