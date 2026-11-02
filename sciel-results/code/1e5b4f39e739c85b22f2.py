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
    
    def residual(x):
        # Reshape x into U, V, W
        U_flat = x[0 : target_rank * dim_U]
        V_flat = x[target_rank * dim_U : target_rank * dim_U + target_rank * dim_V]
        W_flat = x[target_rank * dim_U + target_rank * dim_V : ]
        
        U = U_flat.reshape(target_rank, dim_U)
        V = V_flat.reshape(target_rank, dim_V)
        W = W_flat.reshape(target_rank, dim_W)
        
        # Compute tensor sum
        U_3d = U.T.reshape(n, m, target_rank)
        V_3d = V.T.reshape(m, p, target_rank)
        W_3d = W.T.reshape(p, n, target_rank)
        
        tensor_sum = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Residual
        res = tensor_sum - T
        return res.flatten()
    
    def try_rank_119(x0, max_nfev=3000):
        """Try to solve for rank 119 from initial guess x0."""
        try:
            res = least_squares(residual, x0, method='lm', max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            
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
                
                if error < 1e-8:
                    # Try to round to half-integers with small perturbation search
                    best_rounded_error = float('inf')
                    best_rounded = None
                    
                    # Try direct rounding
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    
                    U_3d_h = U_half.T.reshape(n, m, target_rank)
                    V_3d_h = V_half.T.reshape(m, p, target_rank)
                    W_3d_h = W_half.T.reshape(p, n, target_rank)
                    tensor_sum_h = np.einsum('ijr,jkr,kir->ijk', U_3d_h, V_3d_h, W_3d_h)
                    error_h = np.linalg.norm(tensor_sum_h - T)
                    
                    if error_h < 1e-8:
                        return U_half, V_half, W_half, error_h
                    
                    # Try small perturbations around the solution
                    rng = np.random.RandomState(42)
                    for _ in range(50):
                        perturbation_scale = 0.01 * (1 + rng.random())
                        U_pert = U + rng.normal(0, perturbation_scale, U.shape)
                        V_pert = V + rng.normal(0, perturbation_scale, V.shape)
                        W_pert = W + rng.normal(0, perturbation_scale, W.shape)
                        
                        U_half_p = np.round(U_pert * 2) / 2
                        V_half_p = np.round(V_pert * 2) / 2
                        W_half_p = np.round(W_pert * 2) / 2
                        
                        U_3d_hp = U_half_p.T.reshape(n, m, target_rank)
                        V_3d_hp = V_half_p.T.reshape(m, p, target_rank)
                        W_3d_hp = W_half_p.T.reshape(p, n, target_rank)
                        tensor_sum_hp = np.einsum('ijr,jkr,kir->ijk', U_3d_hp, V_3d_hp, W_3d_hp)
                        error_hp = np.linalg.norm(tensor_sum_hp - T)
                        
                        if error_hp < best_rounded_error:
                            best_rounded_error = error_hp
                            best_rounded = (U_half_p, V_half_p, W_half_p)
                        
                        if error_hp < 1e-8:
                            return U_half_p, V_half_p, W_half_p, error_hp
                    
                    if best_rounded is not None and best_rounded_error < 1e-8:
                        return best_rounded[0], best_rounded[1], best_rounded[2], best_rounded_error
        except Exception:
            pass
        return None
    
    # Multiple restart strategies
    rng = np.random.RandomState(123)
    
    # Strategy 1: Schoolbook initialization with random noise
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        
        x0 = np.zeros(num_params)
        # Fill U part with schoolbook pattern
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
        
        # Add random noise
        noise_scale = 0.1 * (1 + attempt * 0.5)
        x0 += rng.normal(0, noise_scale, num_params)
        
        result = try_rank_119(x0, max_nfev=2000)
        if result is not None:
            best_U = result[0].tolist()
            best_V = result[1].tolist()
            best_W = result[2].tolist()
            best_rank = target_rank
            return {"U": best_U, "V": best_V, "W": best_W}
    
    # Strategy 2: Random initialization
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        
        x0 = rng.normal(0, 1.0, num_params)
        
        result = try_rank_119(x0, max_nfev=3000)
        if result is not None:
            best_U = result[0].tolist()
            best_V = result[1].tolist()
            best_W = result[2].tolist()
            best_rank = target_rank
            return {"U": best_U, "V": best_V, "W": best_W}
    
    # If we didn't find a valid 119, return 120
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
