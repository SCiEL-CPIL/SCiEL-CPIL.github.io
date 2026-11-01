import numpy as np
import time
from scipy.optimize import fsolve

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Target tensor is sum of E_{i,j} \otimes E_{j,k} \otimes E_{k,i}
    # Indices for U: i*m + j
    # Indices for V: j*p + k
    # Indices for W: k*n + i
    
    # We want to find U, V, W of shapes (n*m, R), (m*p, R), (p*n, R)
    # such that sum_r U[:,r] \otimes V[:,r] \otimes W[:,r] = Target
    
    # The target tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # Note: The indices i,j,k are coupled.
    # Specifically, for a fixed (i,j,k), the entry is at (i*m+j, j*p+k, k*n+i).
    # There are n*m*p = 12 such entries.
    
    def get_target_entries():
        entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    entries.append((u_idx, v_idx, w_idx, 1.0))
        return entries
    
    target_entries = get_target_entries()
    
    def residual(params, R):
        # params is a flat array of length R * (n*m + m*p + p*n)
        # Split into U, V, W
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        U = params[0:num_u * R].reshape(num_u, R)
        V = params[num_u * R:num_u * R + num_v * R].reshape(num_v, R)
        W = params[num_u * R + num_v * R:].reshape(num_w, R)
        
        res = np.zeros(len(target_entries))
        for idx, (u_idx, v_idx, w_idx, val) in enumerate(target_entries):
            # Compute sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            s = 0.0
            for r in range(R):
                s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            res[idx] = s - val
        return res
    
    # Try ranks from 7 upwards
    for R in range(7, 13):
        if time.time() > deadline:
            break
            
        # We have 12 equations.
        # We can fix some variables to reduce dimensionality or just use fsolve on all.
        # Total variables: R * (4 + 6 + 6) = 16R.
        # For R=7, 112 variables.
        
        # To help fsolve, we can fix the first column of U to be 1s?
        # No, that might bias it.
        # Let's try a random initialization.
        
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + attempt)
            # Initialize with small random values
            size = R * (n * m + m * p + p * n)
            x0 = np.random.randn(size) * 0.1
            
            # Scale up to avoid zero solution
            x0 = x0 * 2.0
            
            sol, info, ier, msg = fsolve(residual, x0, args=(R,), full_output=True, xtol=1e-8, maxfev=10000)
            
            if ier == 1:
                # Check if residual is close to 0
                res = residual(sol, R)
                if np.linalg.norm(res) < 1e-6:
                    # Extract U, V, W
                    num_u = n * m
                    num_v = m * p
                    num_w = p * n
                    U = sol[0:num_u * R].reshape(num_u, R)
                    V = sol[num_u * R:num_u * R + num_v * R].reshape(num_v, R)
                    W = sol[num_u * R + num_v * R:].reshape(num_w, R)
                    
                    # Try to convert to integers or half-integers
                    # Multiply by 2 to see if they become integers
                    U2 = U * 2
                    V2 = V * 2
                    W2 = W * 2
                    
                    # Check if close to integers
                    if np.allclose(U2, np.round(U2), atol=1e-4) and \
                       np.allclose(V2, np.round(V2), atol=1e-4) and \
                       np.allclose(W2, np.round(W2), atol=1e-4):
                        U_int = np.round(U2) / 2.0
                        V_int = np.round(V2) / 2.0
                        W_int = np.round(W2) / 2.0
                        
                        # Verify exactness
                        res_exact = 0.0
                        for idx, (u_idx, v_idx, w_idx, val) in enumerate(target_entries):
                            s = 0.0
                            for r in range(R):
                                s += U_int[u_idx, r] * V_int[v_idx, r] * W_int[w_idx, r]
                            res_exact += (s - val)**2
                        if res_exact < 1e-9:
                            # Convert to lists
                            U_list = U_int.tolist()
                            V_list = V_int.tolist()
                            W_list = W_int.tolist()
                            return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # If not half-integers, check if integers
                        if np.allclose(U, np.round(U), atol=1e-4) and \
                           np.allclose(V, np.round(V), atol=1e-4) and \
                           np.allclose(W, np.round(W), atol=1e-4):
                            U_int = np.round(U)
                            V_int = np.round(V)
                            W_int = np.round(W)
                            
                            res_exact = 0.0
                            for idx, (u_idx, v_idx, w_idx, val) in enumerate(target_entries):
                                s = 0.0
                                for r in range(R):
                                    s += U_int[u_idx, r] * V_int[v_idx, r] * W_int[w_idx, r]
                                res_exact += (s - val)**2
                            if res_exact < 1e-9:
                                U_list = U_int.tolist()
                                V_list = V_int.tolist()
                                W_list = W_int.tolist()
                                return {"U": U_list, "V": V_list, "W": W_list}
            
            # If fsolve failed or didn't yield clean numbers, try next attempt
    
    # Fallback to schoolbook (Rank 12)
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
