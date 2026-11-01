"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Create the schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    result_schoolbook = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

    # Check for time limit
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer

    # Try to find a lower rank
    # The target tensor T is defined by T[a,b,c] = 1 if indices are consistent
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # Consistency:
    # i_a = a // m, j_a = a % m
    # j_b = b // p, k_b = b % p
    # k_c = c // n, i_c = c % n
    # Match if i_a == i_c and j_a == j_b and k_b == k_c

    def check_consistency(a, b, c):
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        return (i_a == i_c) and (j_a == j_b) and (k_b == k_c)

    # Precompute target tensor indices
    # Total size: (n*m) * (m*p) * (p*n) = 6 * 21 * 14 = 1764
    # This is small enough to store
    size_a = n * m
    size_b = m * p
    size_c = p * n
    total_size = size_a * size_b * size_c
    
    target_tensor = np.zeros(total_size, dtype=float)
    for a in range(size_a):
        for b in range(size_b):
            for c in range(size_c):
                if check_consistency(a, b, c):
                    idx = a * size_b * size_c + b * size_c + c
                    target_tensor[idx] = 1.0

    def residual_func(params, R):
        # params is flattened U, V, W
        # U: (size_a, R), V: (size_b, R), W: (size_c, R)
        num_params_U = size_a * R
        num_params_V = size_b * R
        num_params_W = size_c * R
        
        U_flat = params[:num_params_U]
        V_flat = params[num_params_U:num_params_U + num_params_V]
        W_flat = params[num_params_U + num_params_V:]
        
        U = U_flat.reshape((size_a, R))
        V = V_flat.reshape((size_b, R))
        W = W_flat.reshape((size_c, R))
        
        # Compute tensor product sum
        # sum_r U[:,r] @ V[:,r] @ W[:,r]
        # This is a 3-way tensor sum.
        # We can compute it efficiently using einsum or loops.
        # result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Using broadcasting might be memory intensive if R is large, but R is small.
        # U[a,r] * V[b,r] * W[c,r]
        # Shape: (size_a, size_b, size_c)
        
        # Efficient way:
        # tmp1[a,b,r] = U[a,r] * V[b,r]
        # result[a,b,c] = sum_r tmp1[a,b,r] * W[c,r]
        
        # Let's use einsum
        # U: (a, r), V: (b, r), W: (c, r)
        # result: (a, b, c) = sum_r U[a,r] V[b,r] W[c,r]
        
        # np.einsum('ar,br,cr->abc', U, V, W)
        
        try:
            computed = np.einsum('ar,br,cr->abc', U, V, W)
        except Exception:
            return 1e10
            
        computed_flat = computed.flatten()
        diff = computed_flat - target_tensor
        return np.sum(diff * diff)

    def try_rank(R, max_restarts=5):
        if time.time() > deadline:
            return None
            
        num_params_U = size_a * R
        num_params_V = size_b * R
        num_params_W = size_c * R
        total_params = num_params_U + num_params_V + num_params_W
        
        best_res = float('inf')
        best_params = None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. Schoolbook has 1s.
            init_scale = 1.0
            params = np.random.normal(0, init_scale, total_params)
            
            # Use scipy.optimize.minimize
            from scipy.optimize import minimize
            
            # We need to handle the non-smoothness or just use a robust method.
            # L-BFGS-B is good.
            # We can also try Nelder-Mead if dimensions are small, but 1681 is large.
            
            # To speed up, we can use a coarser tolerance first.
            try:
                res = minimize(
                    lambda x: residual_func(x, R),
                    params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4}
                )
            except Exception:
                continue
                
            if res.fun < best_res:
                best_res = res.fun
                best_params = res.x
                
            # If we found a very good solution, try to polish
            if best_res < 1e-4:
                # Polish with tighter tolerance
                try:
                    res2 = minimize(
                        lambda x: residual_func(x, R),
                        best_params,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res2.fun < best_res:
                        best_res = res2.fun
                        best_params = res2.x
                except Exception:
                    pass
                
                if best_res < 1e-8:
                    # Check for exact integer/half-integer solution
                    # Snap to nearest 0.5
                    U_flat = best_params[:num_params_U]
                    V_flat = best_params[num_params_U:num_params_U + num_params_V]
                    W_flat = best_params[num_params_U + num_params_V:]
                    
                    U_snap = np.round(U_flat * 2) / 2.0
                    V_snap = np.round(V_flat * 2) / 2.0
                    W_snap = np.round(W_flat * 2) / 2.0
                    
                    # Verify exactness with snapped values
                    # Reconstruct and check residual
                    U_s = U_snap.reshape((size_a, R))
                    V_s = V_snap.reshape((size_b, R))
                    W_s = W_snap.reshape((size_c, R))
                    
                    try:
                        computed_snap = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
                        computed_snap_flat = computed_snap.flatten()
                        diff_snap = computed_snap_flat - target_tensor
                        if np.max(np.abs(diff_snap)) < 1e-9:
                            # Found exact solution!
                            return {
                                "U": U_s.tolist(),
                                "V": V_s.tolist(),
                                "W": W_s.tolist()
                            }
                    except Exception:
                        pass
                        
        return None

    # Search for lower ranks
    for R in range(41, 34, -1):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result is not None:
            return result
            
    return result_schoolbook
# EVOLVE-BLOCK-END
