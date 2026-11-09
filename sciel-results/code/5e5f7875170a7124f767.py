import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Create the target tensor T as a flattened vector
    # T[a, b, c] is 1 if a//m == c%n, b//p == a%m, c//n == b%p
    # Indices:
    # a in [0, n*m)
    # b in [0, m*p)
    # c in [0, p*n)
    
    T = np.zeros(dim_u * dim_v * dim_w)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = (a * dim_v + b) * dim_w + c
                T[idx] = 1.0
                
    T_flat = T.flatten()
    
    def compute_residuals(params, R):
        # params: flattened U, V, W
        # U: dim_u x R
        # V: dim_v x R
        # W: dim_w x R
        u_size = dim_u * R
        v_size = dim_v * R
        w_size = dim_w * R
        
        U = params[:u_size].reshape(dim_u, R)
        V = params[u_size:u_size+v_size].reshape(dim_v, R)
        W = params[u_size+v_size:u_size+v_size+w_size].reshape(dim_w, R)
        
        # Compute sum_r U[a,r] V[b,r] W[c,r]
        # This is the tensor product sum.
        # We can compute this efficiently.
        # Residuals are for each (a,b,c)
        # T_decomposed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Vectorized approach:
        # Create a matrix M_ab where M_ab[a, b] = sum_r U[a,r] V[b,r] ? No, that's not right.
        # We need the full 3D tensor.
        # Given small dimensions, we can just loop or use einsum.
        
        # T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        # This might be slow if done many times.
        # Alternative:
        # For each r, outer product of U[:,r], V[:,r], W[:,r]
        
        T_decomp = np.zeros_like(T)
        for r in range(R):
            T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        return T_decomp.flatten() - T_flat

    def try_rank(R, max_restarts=20):
        best_params = None
        best_error = float('inf')
        
        for restart in range(max_restarts):
            # Random initialization
            # Use small integers/half-integers
            # Scale to 1.0
            U_init = np.random.choice([-1, 0, 1], size=(dim_u, R)).astype(float)
            V_init = np.random.choice([-1, 0, 1], size=(dim_v, R)).astype(float)
            W_init = np.random.choice([-1, 0, 1], size=(dim_w, R)).astype(float)
            
            # Normalize to avoid large values
            U_init /= np.linalg.norm(U_init) if np.linalg.norm(U_init) > 0 else 1
            V_init /= np.linalg.norm(V_init) if np.linalg.norm(V_init) > 0 else 1
            W_init /= np.linalg.norm(W_init) if np.linalg.norm(W_init) > 0 else 1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(compute_residuals, params_init, args=(R,), method='lm', max_nfev=1000)
                if res.success:
                    error = np.linalg.norm(res.fun)
                    if error < best_error:
                        best_error = error
                        best_params = res.x
                        if error < 1e-8:
                            break
            except Exception as e:
                continue
        
        if best_params is not None and best_error < 1e-6:
            # Extract U, V, W
            u_size = dim_u * R
            v_size = dim_v * R
            w_size = dim_w * R
            U = best_params[:u_size].reshape(dim_u, R)
            V = best_params[u_size:u_size+v_size].reshape(dim_v, R)
            W = best_params[u_size+v_size:u_size+v_size+w_size].reshape(dim_w, R)
            
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify exactness
            T_decomp = np.zeros_like(T)
            for r in range(R):
                T_decomp += np.einsum('a,b,c->abc', U_rounded[:,r], V_rounded[:,r], W_rounded[:,r])
            
            if np.allclose(T_decomp.flatten(), T_flat, atol=1e-9):
                # Convert to list of lists for return
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                
        return None

    # Try ranks in increasing order of preference (lower is better)
    # Start from 20, go up to 29
    for R in range(20, 30):
        result = try_rank(R, max_restarts=10)
        if result is not None:
            return result
            
    # Fallback to schoolbook
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
