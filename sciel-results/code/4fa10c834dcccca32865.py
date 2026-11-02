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
    deadline = 140  # 10s safety margin
    
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
    
    # Initialize with schoolbook terms 0 to target_rank-1
    x0 = np.zeros(num_params)
    # Fill U part
    for r in range(target_rank):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        U_idx = i * m + j
        V_idx = j * p + k
        W_idx = k * n + i
        x0[r * (dim_U + dim_V + dim_W) + U_idx] = 1.0
        x0[r * (dim_U + dim_V + dim_W) + dim_U + V_idx] = 1.0
        x0[r * (dim_U + dim_V + dim_W) + dim_U + dim_V + W_idx] = 1.0
        
    def residual(x):
        # Reshape x into U, V, W
        # x is flat: [U_0, U_1, ..., U_{R-1}, V_0, ..., V_{R-1}, W_0, ..., W_{R-1}]
        # Each U_r is size dim_U, etc.
        # Actually, it's easier to interleave or block.
        # Let's block: U (R x dim_U), V (R x dim_V), W (R x dim_W)
        # x[0 : R*dim_U] -> U
        # x[R*dim_U : R*dim_U + R*dim_V] -> V
        # x[R*dim_U + R*dim_V : ] -> W
        
        U_flat = x[0 : target_rank * dim_U]
        V_flat = x[target_rank * dim_U : target_rank * dim_U + target_rank * dim_V]
        W_flat = x[target_rank * dim_U + target_rank * dim_V : ]
        
        U = U_flat.reshape(target_rank, dim_U)
        V = V_flat.reshape(target_rank, dim_V)
        W = W_flat.reshape(target_rank, dim_W)
        
        # Compute tensor sum
        # U[r, i*m+j], V[r, j*p+k], W[r, k*n+i]
        # Sum over r
        # Reshape to 3D for einsum
        U_3d = U.T.reshape(n, m, target_rank) # n x m x R
        V_3d = V.T.reshape(m, p, target_rank) # m x p x R
        W_3d = W.T.reshape(p, n, target_rank) # p x n x R
        
        tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Residual
        res = tensor_sum - T
        return res.flatten()
    
    # Run optimization
    try:
        # Use 'lm' if possible, else 'trf'
        # 'lm' is Levenberg-Marquardt, good for least squares
        # But 'lm' doesn't support bounds. We don't need bounds.
        res = least_squares(residual, x0, method='lm', max_nfev=2000)
        
        if res.success or res.cost < 1e-10:
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
                else:
                    # If rounding fails, keep float? No, must be int/half-int.
                    # So we must fail or find another way.
                    # But often the solution IS half-integer.
                    pass
    except Exception:
        pass
    
    # If we didn't find a valid 119, return 120
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
