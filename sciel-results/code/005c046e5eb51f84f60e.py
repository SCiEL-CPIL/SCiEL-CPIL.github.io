import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    rank_target = 28 # Known rank for 2x3x5
    
    # Build the target tensor T as a flat array
    # Dimensions: (n*m, m*p, p*n)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # T_flat[i, j, k] where i in [0, dim_u), j in [0, dim_v), k in [0, dim_w)
    # We represent T as a 3D array for easier manipulation in the objective
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    
    # Populate T based on the multiplication tensor definition
    # T[i, j, k] = 1 if indices are consistent
    # i = i_u * m + j_u
    # j = j_v * p + k_v
    # k = k_w * n + i_w
    # Consistency: j_u == j_v and k_v == k_w and i_w == i_u
    for i_u in range(n):
        for j_u in range(m):
            for k_v in range(p):
                # i index in U
                idx_u = i_u * m + j_u
                # j index in V
                idx_v = j_u * p + k_v
                # k index in W
                idx_w = k_v * n + i_u
                T[idx_u, idx_v, idx_w] = 1.0

    # Function to compute the tensor from U, V, W
    def tensor_from_factors(flat_params):
        # flat_params contains U, V, W stacked
        # U: (dim_u, rank), V: (dim_v, rank), W: (dim_w, rank)
        size_u = dim_u * rank_target
        size_v = dim_v * rank_target
        
        U_flat = flat_params[:size_u]
        V_flat = flat_params[size_u:size_u + size_v]
        W_flat = flat_params[size_u + size_v:]
        
        U = U_flat.reshape((dim_u, rank_target))
        V = V_flat.reshape((dim_v, rank_target))
        W = W_flat.reshape((dim_w, rank_target))
        
        # Compute sum of outer products
        # T_calc[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # This is a 3-mode tensor product
        # We can do this efficiently by iterating over r or using einsum
        # np.einsum('ir,jr,kr->ijk', U, V, W)
        T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
        return T_calc

    def objective(flat_params):
        T_calc = tensor_from_factors(flat_params)
        return (T_calc - T).flatten()

    # Try to find a rank-28 decomposition
    start_time = time.time()
    deadline = start_time + 100 # Leave some margin
    
    best_result = None
    best_error = np.inf
    
    # Random restarts
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Initialize with random values
        # Scaling is important
        scale = 1.0
        params = np.random.randn(dim_u * rank_target + dim_v * rank_target + dim_w * rank_target) * scale
        
        try:
            res = least_squares(objective, params, method='trf', max_nfev=1000)
            if res.cost < best_error:
                best_error = res.cost
                best_result = res.x
        except Exception:
            pass

        # If we get a very small error, we might have a solution
        if best_error < 1e-9:
            break

    # Check if we found a valid solution
    if best_result is not None and best_error < 1e-6:
        # Extract U, V, W
        size_u = dim_u * rank_target
        size_v = dim_v * rank_target
        
        U_flat = best_result[:size_u]
        V_flat = best_result[size_u:size_u + size_v]
        W_flat = best_result[size_u + size_v:]
        
        U = U_flat.reshape((dim_u, rank_target))
        V = V_flat.reshape((dim_v, rank_target))
        W = W_flat.reshape((dim_w, rank_target))
        
        # Round to nearest half-integer or integer to satisfy "integer/half-integer entries"
        # The problem states entries are integer/half-integer.
        # We can multiply by 2, round, then divide by 2.
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify the rounded solution
        T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        if np.allclose(T_check, T, atol=1e-9):
            # Convert to list of lists for the interface
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }

    # Fallback to schoolbook decomposition
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
