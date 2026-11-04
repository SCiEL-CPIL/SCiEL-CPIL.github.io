import numpy as np
import time
import sys
from scipy.optimize import least_squares

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
                
    def verify(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        r = U.shape[1]
        # Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-5:
                        return False
        # Check half-integer constraint
        for mat in [U, V, W]:
            for val in mat:
                if abs(val * 2 - round(val * 2)) > 1e-5:
                    return False
        return True

    def get_residuals(x, r):
        # x is flat array of length (n*m + m*p + p*n) * r
        # Reshape into U, V, W
        U_flat = x[:n*m*r]
        V_flat = x[n*m*r : n*m*r + m*p*r]
        W_flat = x[n*m*r + m*p*r :]
        
        U = U_flat.reshape((n*m, r))
        V = V_flat.reshape((m*p, r))
        W = W_flat.reshape((p*n, r))
        
        # Calculate residuals for all i, j, k
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    residuals.append(val - 1.0)
        return np.array(residuals)

    def try_find_decomposition(target_rank, time_limit):
        start_time = time.time()
        num_vars = (n*m + m*p + p*n) * target_rank
        bounds = (-2.0, 2.0) # Half integers likely in this range
        
        np.random.seed(42)
        
        # Multiple restarts
        while time.time() - start_time < time_limit:
            # Initialization strategy
            # 1. Random small values
            # 2. Perturbed schoolbook (if dimensions allow, here we need to reduce rank so we take first 'target_rank' cols and perturb)
            
            # Strategy: Take schoolbook, drop last column, perturb the rest
            if target_rank < rank_schoolbook:
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                # Copy first target_rank columns of schoolbook
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < target_rank:
                                U_init[i * m + j, col] = U_sb[i * m + j, col]
                                V_init[j * p + k, col] = V_sb[j * p + k, col]
                                W_init[k * n + i, col] = W_sb[k * n + i, col]
                                col += 1
                
                # Perturb to allow optimization to move away from singularities
                U_init += np.random.randn(*U_init.shape) * 0.1
                V_init += np.random.randn(*V_init.shape) * 0.1
                W_init += np.random.randn(*W_init.shape) * 0.1
            else:
                U_init = np.random.randn(n*m, target_rank) * 0.1
                V_init = np.random.randn(m*p, target_rank) * 0.1
                W_init = np.random.randn(p*n, target_rank) * 0.1

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use least_squares
                res = least_squares(get_residuals, x0, args=(target_rank,), method='trf', bounds=bounds, max_nfev=1000)
                
                if res.cost < 1e-8:
                    # Extract solution
                    U_flat = res.x[:n*m*target_rank]
                    V_flat = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                    W_flat = res.x[n*m*target_rank + m*p*target_rank :]
                    
                    U_sol = U_flat.reshape((n*m, target_rank))
                    V_sol = V_flat.reshape((m*p, target_rank))
                    W_sol = W_flat.reshape((p*n, target_rank))
                    
                    # Snap to half-integers
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    if verify(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
                        
            except Exception:
                pass
            
        return None

    # Try to find rank 23
    # Time budget: 120s total. Leave 10s for cleanup/return.
    # Try rank 23 with 100s
    sol_23 = try_find_decomposition(23, 100)
    
    if sol_23 is not None:
        U, V, W = sol_23
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Return schoolbook rank 24
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
