import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 120
    
    # Deadline guard
    start_time = time.time()
    deadline = 140  # 10s safety margin before 150s kill
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = rank_school
    
    # Target tensor T[i,j,k] = 1
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Try to find rank 119
    target_rank = 119
    
    # Dimensions for U, V, W
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Total number of parameters
    num_params = target_rank * (dim_U + dim_V + dim_W)
    
    # Pre-compute schoolbook indices
    schoolbook_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_indices.append((i, j, k))
                
    def residual(x):
        # x is flat: [U_0, ..., U_{R-1}, V_0, ..., V_{R-1}, W_0, ..., W_{R-1}]
        # Each U_r is size dim_U, etc.
        # Block: U (R x dim_U), V (R x dim_V), W (R x dim_W)
        
        U_flat = x[0 : target_rank * dim_U]
        V_flat = x[target_rank * dim_U : target_rank * dim_U + target_rank * dim_V]
        W_flat = x[target_rank * dim_U + target_rank * dim_V : ]
        
        U = U_flat.reshape(target_rank, dim_U)
        V = V_flat.reshape(target_rank, dim_V)
        W = W_flat.reshape(target_rank, dim_W)
        
        # Compute tensor sum
        U_3d = U.T.reshape(n, m, target_rank) # n x m x R
        V_3d = V.T.reshape(m, p, target_rank) # m x p x R
        W_3d = W.T.reshape(p, n, target_rank) # p x n x R
        
        tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Residual
        res = tensor_sum - T
        return res.flatten()
    
    # Attempt to find rank 119 decomposition
    # Strategy: Initialize with 118 schoolbook terms and optimize the 119th
    # We will try a few different initializations for the 119th term
    
    success = False
    
    for attempt in range(5): # Try up to 5 times if time permits
        if time.time() > deadline:
            break
            
        x0 = np.zeros(num_params)
        
        # Fill first 118 terms with schoolbook values
        for r in range(118):
            i, j, k = schoolbook_indices[r]
            U_idx = i * m + j
            V_idx = j * p + k
            W_idx = k * n + i
            
            x0[r * (dim_U + dim_V + dim_W) + U_idx] = 1.0
            x0[r * (dim_U + dim_V + dim_W) + dim_U + V_idx] = 1.0
            x0[r * (dim_U + dim_V + dim_W) + dim_U + dim_V + W_idx] = 1.0
        
        # Initialize the 119th term
        # The remaining schoolbook terms are indices 118 and 119
        # We try to approximate their sum with a single rank-1 term
        # This is a heuristic initialization
        
        i1, j1, k1 = schoolbook_indices[118]
        i2, j2, k2 = schoolbook_indices[119]
        
        # Average the indices? No, indices are discrete.
        # We initialize the 119th term to be the sum of the two vectors, normalized
        # This might not be a good rank-1 approximation, but it's a start.
        
        U_init = np.zeros(dim_U)
        V_init = np.zeros(dim_V)
        W_init = np.zeros(dim_W)
        
        U_init[i1 * m + j1] = 1.0
        U_init[i2 * m + j2] = 1.0
        
        V_init[j1 * p + k1] = 1.0
        V_init[j2 * p + k2] = 1.0
        
        W_init[k1 * n + i1] = 1.0
        W_init[k2 * n + i2] = 1.0
        
        # Normalize to keep magnitudes reasonable
        norm_U = np.linalg.norm(U_init)
        norm_V = np.linalg.norm(V_init)
        norm_W = np.linalg.norm(W_init)
        
        if norm_U > 0: U_init /= norm_U
        if norm_V > 0: V_init /= norm_V
        if norm_W > 0: W_init /= norm_W
        
        # Scale factor to match the magnitude (sum of two terms is magnitude ~2)
        # The product of norms is ~1, so we scale by 2?
        # Actually, the term is u v w. The sum of two terms is u1 v1 w1 + u2 v2 w2.
        # If we use u = u1+u2, v=v1+v2, w=w1+w2, the product is (u1+u2)(v1+v2)(w1+w2)
        # which includes cross terms.
        # Let's just set it to the sum and let the optimizer sort it out.
        
        x0[118 * (dim_U + dim_V + dim_W) : 118 * (dim_U + dim_V + dim_W) + dim_U] = U_init
        x0[118 * (dim_U + dim_V + dim_W) + dim_U : 118 * (dim_U + dim_V + dim_W) + dim_U + dim_V] = V_init
        x0[118 * (dim_U + dim_V + dim_W) + dim_U + dim_V : ] = W_init
        
        # Run optimization
        try:
            res = least_squares(residual, x0, method='lm', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            
            if res.cost < 1e-10:
                # Extract solution
                U_flat = res.x[0 : target_rank * dim_U]
                V_flat = res.x[target_rank * dim_U : target_rank * dim_U + target_rank * dim_V]
                W_flat = res.x[target_rank * dim_U + target_rank * dim_V : ]
                
                U = U_flat.reshape(dim_U, target_rank)
                V = V_flat.reshape(dim_V, target_rank)
                W = W_flat.reshape(dim_W, target_rank)
                
                # Check error
                U_3d = U.T.reshape(n, m, target_rank)
                V_3d = V.T.reshape(m, p, target_rank)
                W_3d = W.T.reshape(p, n, target_rank)
                tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                error = np.linalg.norm(tensor_sum - T)
                
                if error < 1e-9:
                    # Try to round to half-integers
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    
                    U_3d_h = U_half.T.reshape(n, m, target_rank)
                    V_3d_h = V_half.T.reshape(m, p, target_rank)
                    W_3d_h = W_half.T.reshape(p, n, target_rank)
                    tensor_sum_h = np.einsum('ijr,jkr,kir->ijk', U_3d_h, V_3d_h, W_3d_h)
                    error_h = np.linalg.norm(tensor_sum_h - T)
                    
                    if error_h < 1e-9:
                        best_U = U_half.tolist()
                        best_V = V_half.tolist()
                        best_W = W_half.tolist()
                        best_rank = target_rank
                        success = True
                        break
        except Exception:
            pass
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
