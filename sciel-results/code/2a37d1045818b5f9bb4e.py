import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Target Tensor T
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Try to find a decomposition of lower rank
    # Start with R=20 (nm + mp - 1)
    # Then try 19, 18... down to some limit
    
    best_result = None
    start_time = time.time()
    time_limit = 110.0 # 10s margin
    
    # We will try ranks from 20 down to 10
    # If we find one, we return it.
    # To save time, we only do a few restarts for each rank.
    
    for R in range(20, 9, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find solution for rank R
        found = False
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W
            # Random initialization
            np.random.seed(42 + R + restart)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Flatten
            def flatten(x):
                return x.flatten()
            
            def unflatten(x, shape):
                return x.reshape(shape)
                
            # Objective function
            def loss(params):
                # params is flattened U, V, W
                # Sizes: nm*R, mp*R, pn*R
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                
                U_flat = params[:size_U]
                V_flat = params[size_U:size_U+size_V]
                W_flat = params[size_U+size_V:]
                
                U_mat = unflatten(U_flat, (n * m, R))
                V_mat = unflatten(V_flat, (m * p, R))
                W_mat = unflatten(W_flat, (p * n, R))
                
                # Compute tensor product sum
                # Sum_r U[:, r] outer V[:, r] outer W[:, r]
                # This is expensive if done naively.
                # T_recon = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
                # But T is sparse. We only need to check non-zero entries?
                # No, we need to minimize error over all entries.
                # However, T is 0 almost everywhere.
                # We can just compute the full tensor and subtract T.
                
                T_recon = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
                error = T_recon - T
                return np.sum(error * error)
                
            # Use scipy.optimize
            try:
                from scipy.optimize import minimize
                x0 = np.concatenate([flatten(U), flatten(V), flatten(W)])
                
                # Use L-BFGS-B
                res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
                
                if res.success or res.fun < 1e-6:
                    # Check if we can round to integers/half-integers
                    U_opt = unflatten(res.x[:size_U], (n * m, R))
                    V_opt = unflatten(res.x[size_U:size_U+size_V], (m * p, R))
                    W_opt = unflatten(res.x[size_U+size_V:], (p * n, R))
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    T_recon = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
                    if np.allclose(T_recon, T, atol=1e-9):
                        # Valid solution found
                        # Convert to lists
                        best_result = {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    if best_result is not None:
        return best_result
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

# EVOLVE-BLOCK-END
