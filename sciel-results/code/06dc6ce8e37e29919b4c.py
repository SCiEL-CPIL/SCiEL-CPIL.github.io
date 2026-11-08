"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x10 matrix.
    
    First tries to find a rank-54 decomposition via numerical optimization.
    Falls back to the known rank-55 decomposition if that fails.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 30
    dim_w = p * n  # 20

    def build_tensor():
        """Build the target multiplication tensor T[i,j,k] = 1."""
        T = np.ones((dim_u, dim_v, dim_w))
        return T

    def tensor_error(U, V, W):
        """Compute the Frobenius norm of the tensor reconstruction error."""
        # T_recon[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Using einsum
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.linalg.norm(T_recon - np.ones((dim_u, dim_v, dim_w)))

    def try_rank(R, max_iters=50, timeout=30.0):
        """Try to find a rank-R decomposition via numerical optimization."""
        if time.time() > deadline - 10:
            return None
        
        best_result = None
        best_err = float('inf')
        
        for attempt in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            np.random.seed(attempt)
            U = np.random.randn(dim_u, R) * 0.5
            V = np.random.randn(dim_v, R) * 0.5
            W = np.random.randn(dim_w, R) * 0.5
            
            # Scale to match the tensor magnitude
            scale = (dim_u * dim_v * dim_w) ** (1.0 / 6.0)
            U *= scale
            V *= scale
            W *= scale
            
            err = tensor_error(U, V, W)
            
            # Simple gradient-free optimization using random perturbations
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Random perturbation
                delta = np.random.randn(dim_u, R) * 0.01
                U_new = U + delta
                err_new = tensor_error(U_new, V, W)
                if err_new < err:
                    U = U_new
                    err = err_new
                
                delta = np.random.randn(dim_v, R) * 0.01
                V_new = V + delta
                err_new = tensor_error(U, V_new, W)
                if err_new < err:
                    V = V_new
                    err = err_new
                
                delta = np.random.randn(dim_w, R) * 0.01
                W_new = W + delta
                err_new = tensor_error(U, V, W_new)
                if err_new < err:
                    W = W_new
                    err = err_new
                
                if err < 1e-10:
                    break
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-10:
                break
        
        if best_result is not None and best_err < 1e-10:
            return best_result
        return None

    # Try rank 54
    result_54 = None
    if time.time() < deadline - 15:
        result_54 = try_rank(54, max_iters=3, timeout=30.0)
    
    if result_54 is not None:
        U, V, W = result_54
        # Convert to lists with rounded values
        U_list = [[round(U[i, r], 6) for r in range(54)] for i in range(dim_u)]
        V_list = [[round(V[i, r], 6) for r in range(54)] for i in range(dim_v)]
        W_list = [[round(W[i, r], 6) for r in range(54)] for i in range(dim_w)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: rank-55 decomposition
    num_blocks = p // 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    strassen_terms = [
        [
            ('U', (0, 0), 1), ('U', (1, 1), 1),
            ('V', (0, 0), 1), ('V', (1, 1), 1),
            ('W', (0, 0), 1), ('W', (1, 1), 1)
        ],
        [
            ('U', (1, 0), 1), ('U', (1, 1), 1),
            ('V', (0, 0), 1),
            ('W', (0, 1), 1), ('W', (1, 1), -1)
        ],
        [
            ('U', (0, 0), 1),
            ('V', (0, 1), 1), ('V', (1, 1), -1),
            ('W', (1, 0), 1), ('W', (1, 1), 1)
        ],
        [
            ('U', (1, 1), 1),
            ('V', (1, 0), 1), ('V', (0, 0), -1),
            ('W', (0, 0), 1), ('W', (0, 1), 1)
        ],
        [
            ('U', (0, 0), 1), ('U', (0, 1), 1),
            ('V', (1, 1), 1),
            ('W', (0, 0), -1), ('W', (1, 0), 1)
        ],
        [
            ('U', (1, 0), 1), ('U', (0, 0), -1),
            ('V', (0, 0), 1), ('V', (0, 1), 1),
            ('W', (1, 1), 1)
        ],
        [
            ('U', (0, 1), 1), ('U', (1, 1), -1),
            ('V', (1, 0), 1), ('V', (1, 1), 1),
            ('W', (0, 0), 1)
        ]
    ]
    
    outer_terms = [
        [
            ('U', (0, 2), 1),
            ('V', (2, 0), 1),
            ('W', (0, 0), 1)
        ],
        [
            ('U', (0, 2), 1),
            ('V', (2, 1), 1),
            ('W', (1, 0), 1)
        ],
        [
            ('U', (1, 2), 1),
            ('V', (2, 0), 1),
            ('W', (0, 1), 1)
        ],
        [
            ('U', (1, 2), 1),
            ('V', (2, 1), 1),
            ('W', (1, 1), 1)
        ]
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        
        for term in all_terms:
            col = col_offset
            for mat_name, idx, val in term:
                if mat_name == 'U':
                    i, j = idx
                    U[i * m + j][col] += val
                elif mat_name == 'V':
                    j, k_local = idx
                    k_global = b * block_size + k_local
                    V[j * p + k_global][col] += val
                elif mat_name == 'W':
                    k_local, i = idx
                    k_global = b * block_size + k_local
                    W[k_global * n + i][col] += val
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
