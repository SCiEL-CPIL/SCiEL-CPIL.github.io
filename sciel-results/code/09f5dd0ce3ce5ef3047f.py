import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 9]

def _build_block_tensor():
    """Build M_{2,3,3} tensor."""
    n, m, p = 2, 3, 3
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _solve_block(rank, T, max_restarts=20, timeout=10.0):
    """Solve for rank decomposition of block tensor."""
    a, b, c = T.shape
    start_time = time.time()
    
    best_residual = float('inf')
    best_params = None
    
    for r in range(max_restarts):
        if time.time() > timeout:
            break
        rng = np.random.RandomState(r * 42)
        # Initialize
        U = rng.randn(a, rank)
        V = rng.randn(b, rank)
        W = rng.randn(c, rank)
        
        # Flatten
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residuals(x):
            U_flat = x[:a*rank]
            V_flat = x[a*rank:a*rank+b*rank]
            W_flat = x[a*rank+b*rank:]
            
            U = U_flat.reshape(a, rank)
            V = V_flat.reshape(b, rank)
            W = W_flat.reshape(c, rank)
            
            # Reconstruct tensor
            # T_recon = sum_r U[:,r] outer V[:,r] outer W[:,r]
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_recon - T).flatten()
        
        try:
            res = least_squares(residuals, x0, method='lm', max_nfev=1000)
            residual_norm = res.cost * 2 # cost is 0.5 * sum(residuals^2)
            if residual_norm < best_residual:
                best_residual = residual_norm
                best_params = res.x
                if residual_norm < 1e-10:
                    break
        except Exception:
            continue
    
    if best_residual < 1e-6:
        x = best_params
        U = x[:a*rank].reshape(a, rank)
        V = x[a*rank:a*rank+b*rank].reshape(b, rank)
        W = x[a*rank+b*rank:].reshape(c, rank)
        return U, V, W, best_residual
    return None, None, None, best_residual

def construct():
    n, m, p = DIMENSIONS
    # Block dimensions
    bn, bm, bp = 2, 3, 3
    num_blocks = p // bp # 9 // 3 = 3
    
    T_block = _build_block_tensor()
    
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Try to find a low rank decomposition for the block
    # Known rank is 7
    best_block_rank = 18 # schoolbook
    best_block_U = None
    best_block_V = None
    best_block_W = None
    
    for r in range(7, 19):
        if time.time() > deadline - 10:
            break
        U_b, V_b, W_b, res = _solve_block(r, T_block, max_restarts=5, timeout=5.0)
        if U_b is not None and res < 1e-6:
            best_block_rank = r
            best_block_U = U_b
            best_block_V = V_b
            best_block_W = W_b
            break
    
    # If we found a block solution, build the full solution
    if best_block_U is not None:
        R = best_block_rank
        total_rank = R * num_blocks
        
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        for b_idx in range(num_blocks):
            col_start = b_idx * R
            col_end = (b_idx + 1) * R
            
            # U is same for all blocks
            U_full[:, col_start:col_end] = best_block_U
            
            # V mapping
            # Block V index: j * 3 + l
            # Full V index: j * 9 + k, where k = 3 * b_idx + l
            for j in range(bm):
                for l in range(bp):
                    v_idx_block = j * bp + l
                    v_idx_full = j * p + (3 * b_idx + l)
                    V_full[v_idx_full, col_start:col_end] = best_block_V[v_idx_block, :]
            
            # W mapping
            # Block W index: i * 3 + l
            # Full W index: i * 9 + k, where k = 3 * b_idx + l
            for i in range(bn):
                for l in range(bp):
                    w_idx_block = i * bp + l
                    w_idx_full = i * p + (3 * b_idx + l)
                    W_full[w_idx_full, col_start:col_end] = best_block_W[w_idx_block, :]
        
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n * m, rank_schoolbook))
        V_sb = np.zeros((m * p, rank_schoolbook))
        W_sb = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
