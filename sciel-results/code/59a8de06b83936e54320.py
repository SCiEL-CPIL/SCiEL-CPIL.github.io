"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def _find_small_matmul_decomposition(n, m, p, max_rank=20):
    """
    Find a rank-minimal decomposition for n x m x p matrix multiplication.
    Returns U, V, W or None if not found.
    """
    # Tensor T[i, j, k] = 1 for all i, j, k
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    num_constraints = n * m * p
    # Flattened indices
    # U has size n*m, V has size m*p, W has size p*n
    
    for rank in range(1, max_rank + 1):
        # Initial guess: random small values
        # Or use a structured guess
        # Let's try a few random restarts
        found = False
        for restart in range(5):
            U0 = np.random.randn(n * m, rank) * 0.1
            V0 = np.random.randn(m * p, rank) * 0.1
            W0 = np.random.randn(p * n, rank) * 0.1
            
            def residuals(params):
                U = params[:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                # Compute tensor decomposition
                # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # We need to check if this equals 1 for all i,j,k
                
                # Efficient computation using einsum or loops
                # Since dimensions are small, loops are fine
                res = np.zeros(num_constraints)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            res[idx] = val - 1.0
                            idx += 1
                return res

            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=2000)
                if result.success or result.cost < 1e-10:
                    U = result.x[:n*m*rank].reshape(n*m, rank)
                    V = result.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                    W = result.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                    
                    # Round to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exactness
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for c in range(rank):
                                    val += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                                err = abs(val - 1.0)
                                if err > max_err:
                                    max_err = err
                                    
                    if max_err < 1e-9:
                        return U_r, V_r, W_r, rank
            except Exception:
                pass
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Decompose 2x3x9 into three 2x3x3 blocks
    # B is 3x9, so B = [B1, B2, B3] where each Bi is 3x3
    # C = A B = [A B1, A B2, A B3]
    # The tensor is the direct sum of three 2x3x3 tensors.
    # So we find a decomposition for 2x3x3 and replicate it.
    
    sub_n, sub_m, sub_p = 2, 3, 3
    num_blocks = 3 # Since p=9 and sub_p=3
    
    # Find decomposition for 2x3x3
    result = _find_small_matmul_decomposition(sub_n, sub_m, sub_p, max_rank=18)
    
    if result is not None:
        U_sub, V_sub, W_sub, rank_sub = result
        total_rank = rank_sub * num_blocks
        
        # Construct full U, V, W
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        for b in range(num_blocks):
            # U is independent of k, so we just repeat U_sub for each block?
            # Wait.
            # The tensor is T[i, j, k].
            # For block b, k' = k % 3.
            # The term for block b is U_sub[i*m+j, c] * V_sub[j*sub_p + k', c] * W_sub[k'*sub_n + i, c]
            # In the full tensor, the index for V is j*p + k.
            # For block b, k = b*3 + k'. So V index is j*9 + b*3 + k'.
            # V_sub index is j*3 + k'.
            # So V_full[j*9 + b*3 + k', b*rank_sub + c] = V_sub[j*3 + k', c]
            
            # W index is k*n + i.
            # For block b, k = b*3 + k'. So W index is (b*3 + k')*2 + i.
            # W_sub index is k'*2 + i.
            # So W_full[(b*3 + k')*2 + i, b*rank_sub + c] = W_sub[k'*2 + i, c]
            
            # U index is i*m + j.
            # U_full[i*m + j, b*rank_sub + c] = U_sub[i*m + j, c]
            
            for i in range(sub_n):
                for j in range(sub_m):
                    for c in range(rank_sub):
                        U[i*sub_m + j, b*rank_sub + c] = U_sub[i*sub_m + j, c]
                        
                        for k_prime in range(sub_p):
                            k_full = b*sub_p + k_prime
                            V[j*p + k_full, b*rank_sub + c] = V_sub[j*sub_p + k_prime, c]
                            W[k_full*sub_n + i, b*rank_sub + c] = W_sub[k_prime*sub_n + i, c]
                            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
