import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n×m×p matrix multiplication tensor."""
    rank = U.shape[1]
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute decomposition tensor
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _find_subproblem_decomposition(n, m, p, R, max_iters=100, seeds=20):
    """
    Find a rank-R decomposition for the n x m x p multiplication tensor.
    Uses scipy.optimize.least_squares.
    """
    # Target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    T_flat = T.flatten()
    
    # Parameters: flatten U, V, W
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    
    def residuals(params):
        U = params[:size_U].reshape(n * m, R)
        V = params[size_U:size_U + size_V].reshape(m * p, R)
        W = params[size_U + size_V:].reshape(p * n, R)
        
        T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
        return T_dec.flatten() - T_flat

    best_solution = None
    best_cost = np.inf

    for seed in range(seeds):
        np.random.seed(seed)
        # Random initialization
        # Scale initialization to avoid large initial errors
        init_U = np.random.randn(n * m, R) * 0.1
        init_V = np.random.randn(m * p, R) * 0.1
        init_W = np.random.randn(p * n, R) * 0.1
        
        # Try a few different scales
        for scale in [0.1, 1.0, 10.0]:
            init_U *= scale
            init_V *= scale
            init_W *= scale
            
            x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=10000)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_solution = result.x
                    if result.cost < 1e-10:
                        break
            except Exception:
                continue
        if best_cost < 1e-10:
            break

    if best_solution is not None and best_cost < 1e-10:
        U = best_solution[:size_U].reshape(n * m, R)
        V = best_solution[size_U:size_U + size_V].reshape(m * p, R)
        W = best_solution[size_U + size_V:].reshape(p * n, R)
        return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Strategy: Decompose into 3 blocks of 2x3x2
    # If we find rank 8 for 2x3x2, total rank is 24.
    n_sub, m_sub, p_sub = 2, 3, 2
    R_sub = 8
    num_blocks = p // p_sub  # 3 blocks
    
    U_full = None
    V_full = None
    W_full = None
    R_full = 0
    
    # Try to find rank 8 decomposition for 2x3x2
    res = _find_subproblem_decomposition(n_sub, m_sub, p_sub, R_sub, seeds=10)
    
    if res is not None:
        U_sub, V_sub, W_sub = res
        # Verify subproblem
        if _verify_decomposition(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub):
            R_full = num_blocks * R_sub
            U_full = np.zeros((n * m, R_full))
            V_full = np.zeros((m * p, R_full))
            W_full = np.zeros((p * n, R_full))
            
            for block in range(num_blocks):
                offset = block * R_sub
                # U is the same for all blocks (A is shared)
                U_full[:, offset:offset+R_sub] = U_sub
                
                # V and W need to be mapped to the specific block columns
                # Block b corresponds to columns [b*p_sub, (b+1)*p_sub) of B and C
                for j in range(m_sub):
                    for k in range(p_sub):
                        # B index: j * p + (block * p_sub + k)
                        # V_sub index: j * p_sub + k
                        V_full[j * p + (block * p_sub + k), offset:offset+R_sub] = V_sub[j * p_sub + k]
                        
                for k in range(p_sub):
                    for i in range(n_sub):
                        # C index: (block * p_sub + k) * n + i
                        # W_sub index: k * n_sub + i
                        W_full[(block * p_sub + k) * n + i, offset:offset+R_sub] = W_sub[k * n_sub + i]
            
            # Verify full decomposition
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                return {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist()
                }
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
