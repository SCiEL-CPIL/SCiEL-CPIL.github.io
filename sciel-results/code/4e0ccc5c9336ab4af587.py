import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find a lower rank
    # The tensor T has non-zero entries at (u, v, w) where u=i*m+j, v=j*p+k, w=k*n+i
    # We want U, V, W such that sum_r U[u,r] V[v,r] W[w,r] == T[u,v,w]
    
    # List of non-zero indices
    non_zero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                non_zero_indices.append((u, v, w))
                
    # Total number of entries in the tensor is (n*m)*(m*p)*(p*n)
    # But we only care about the non-zero ones for the equations?
    # No, the other entries must be 0.
    # So we have (n*m)*(m*p)*(p*n) equations.
    # 6 * 15 * 10 = 900 equations.
    
    # We will search for ranks from 12 to 25
    # Using random restarts and least squares
    
    import scipy.optimize as opt
    import random
    
    def objective(params):
        # params is a flattened array of U, V, W
        # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)
        R = params.size // (n*m + m*p + p*n)
        if R == 0: return 1e10
        
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U = params[:size_U * R].reshape(size_U, R)
        V = params[size_U * R : size_U * R + size_V * R].reshape(size_V, R)
        W = params[size_U * R + size_V * R :].reshape(size_W, R)
        
        # Compute the reconstructed tensor
        # T_recon[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # We need to check against T_target
        # Instead of building the full tensor, we can compute the error on the fly
        # But 900 terms is small.
        
        # Let's build the target tensor as a sparse representation or just a list
        # For speed, we precompute the target values
        # T_target is 1 at non_zero_indices, 0 elsewhere.
        
        # We can compute the sum for each (u, v, w)
        # This is O(R * n * m * m * p * p * n) which is R * 900.
        # For R=20, 18000 ops. Very fast.
        
        # However, doing this in Python loops is slow.
        # Let's use numpy einsum if possible, but the indices are specific.
        
        # Alternative:
        # The condition is that for each r, U_r, V_r, W_r form a rank-1 tensor.
        # The sum of these rank-1 tensors must equal T.
        
        # Let's just compute the error for the non-zero entries and a sample of zero entries?
        # No, we need exactness.
        # Let's compute the full tensor error.
        
        # T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        # This creates a (6, 15, 10) array.
        # Target is 0 except at specific spots.
        
        # To avoid creating the full array repeatedly in the objective function,
        # we can precompute the indices.
        
        # Let's try to be smart:
        # The target tensor T is defined by T[u, v, w] = 1 if u% m == v//p and w%n == u//m and w//n == v%p
        # else 0.
        
        # We can check the error only on the non-zero entries and assume the zeros are satisfied?
        # No, that's dangerous.
        
        # Let's use a vectorized approach for the error calculation.
        # We can precompute the target tensor as a flat array of length 900.
        
        # But creating T_recon takes time.
        # Let's try to optimize the objective function.
        
        # For now, let's just use a simple loop in Python to see if it's fast enough for a few iterations.
        # 900 iterations * R (say 15) = 13500 multiplications.
        # In Python, this might take ~1ms.
        # If we need 1000 iterations, that's 1 second.
        # This is acceptable.
        
        err = 0.0
        # We need to iterate over all u, v, w
        for u in range(n * m):
            i = u // m
            j = u % m
            for v in range(m * p):
                j_v = v // p
                k = v % p
                # Check if j == j_v
                if j != j_v:
                    # Target is 0
                    # Sum_r U[u, r] V[v, r] W[w, r] must be 0
                    # We can't easily sum over w here without looping w.
                    # So we must loop w.
                    pass
                # This nested loop in Python is too slow for optimization.
                
        # We must use numpy.
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Target tensor
        # We precompute target_tensor outside
        # But we can't access it easily inside if it's a closure? Yes we can.
        
        # Let's precompute target_tensor
        # It's (6, 15, 10)
        
        # The error is || T_recon - T_target ||_F
        return np.linalg.norm(T_recon - target_tensor)

    # Precompute target_tensor
    target_tensor = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_tensor[u, v, w] = 1.0

    # Search for rank
    for R in range(12, 26):
        if time.time() > deadline:
            break
            
        size_U = n * m
        size_V = m * p
        size_W = p * n
        total_vars = R * (size_U + size_V + size_W)
        
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. Let's try small random values.
            x0 = np.random.randn(total_vars) * 0.1
            
            # Use Nelder-Mead or L-BFGS-B
            # Nelder-Mead is derivative-free and robust for small problems.
            try:
                res = opt.minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
                
                if res.fun < 1e-9:
                    # Found a solution!
                    # Check if it can be rounded to integers/halves
                    x_opt = res.x
                    U = x_opt[:size_U * R].reshape(size_U, R)
                    V = x_opt[size_U * R : size_U * R + size_V * R].reshape(size_V, R)
                    W = x_opt[size_U * R + size_V * R :].reshape(size_W, R)
                    
                    # Try to round to integers or half-integers
                    # Multiply by 2
                    U2 = np.round(U * 2) / 2
                    V2 = np.round(V * 2) / 2
                    W2 = np.round(W * 2) / 2
                    
                    # Check if the rounded solution is exact
                    T_rounded = np.einsum('ur,vr,wr->uvw', U2, V2, W2)
                    if np.allclose(T_rounded, target_tensor, atol=1e-9):
                        # Convert to list of lists with integer/half-integer values
                        # The prompt asks for integer/half-integer entries.
                        # If U2 has values like 0.5, 1.0, -0.5, it's valid.
                        
                        U_list = [[float(val) for val in row] for row in U2]
                        V_list = [[float(val) for val in row] for row in V2]
                        W_list = [[float(val) for val in row] for row in W2]
                        
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
                    # If rounding failed, try rounding to integers only?
                    U2_int = np.round(U)
                    V2_int = np.round(V)
                    W2_int = np.round(W)
                    T_rounded_int = np.einsum('ur,vr,wr->uvw', U2_int, V2_int, W2_int)
                    if np.allclose(T_rounded_int, target_tensor, atol=1e-9):
                        U_list = [[float(val) for val in row] for row in U2_int]
                        V_list = [[float(val) for val in row] for row in V2_int]
                        W_list = [[float(val) for val in row] for row in W2_int]
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception as e:
                pass

    return best_result

# EVOLVE-BLOCK-END
