"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: Use numerical optimization to find a decomposition of rank R.
    We try ranks starting from a lower bound up to the known bound 44.
    """
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Precompute target tensor T_{i,j,k} = delta_{j,k}
    # We need to satisfy: sum_r W[k,i] * U[i,j] * V[j,k] = delta_{j,k}
    # Note: W index in output is k*n + i.
    
    def error_func(params, R):
        # params shape: (R * (n*m + m*p + p*n),)
        # U: R x n*m
        # V: R x m*p
        # W: R x p*n
        
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U = params[0 : R*size_U].reshape(R, size_U)
        V = params[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
        W = params[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
        
        # Residuals
        # For each i, j, k:
        # val = sum_r W[r, k*n+i] * U[r, i*m+j] * V[r, j*p+k]
        # target = 1 if j==k else 0
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum
                    # W_idx = k*n + i
                    # U_idx = i*m + j
                    # V_idx = j*p + k
                    w_col = W[:, k*n + i]
                    u_row = U[:, i*m + j]
                    v_row = V[:, j*p + k]
                    
                    val = np.sum(w_col * u_row * v_row)
                    target = 1.0 if j == k else 0.0
                    residuals.append(val - target)
                    
        return np.array(residuals)

    def try_rank(R, max_restarts=5):
        # Try to find a solution of rank R
        size_U = n * m
        size_V = m * p
        size_W = p * n
        total_size = R * (size_U + size_V + size_W)
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                return None
                
            # Initialize with small random values
            np.random.seed(42 + restart)
            params = np.random.randn(total_size) * 0.1
            
            # Add the known rank 44 solution as a starting point if R >= 44?
            # No, we are trying R < 44.
            # But we can try to initialize with a subset of the rank 44 solution?
            # That might be tricky. Random is safer for structure discovery.
            
            try:
                res = least_squares(error_func, params, args=(R,), method='lm', max_nfev=5000)
            except Exception:
                continue
                
            if res.success and res.cost < 1e-12:
                # Found a solution
                U = res.x[0 : R*size_U].reshape(R, size_U)
                V = res.x[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
                W = res.x[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
                
                # Check if we can snap to integers/half-integers
                # The problem requires integer/half-integer entries.
                # We try snapping to nearest 0.5
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify snapped solution
                residuals_snapped = error_func(
                    np.concatenate([U_snapped.flatten(), V_snapped.flatten(), W_snapped.flatten()]), 
                    R
                )
                
                if np.max(np.abs(residuals_snapped)) < 1e-9:
                    # Convert to lists of lists as required
                    U_list = U_snapped.tolist()
                    V_list = V_snapped.tolist()
                    W_list = W_snapped.tolist()
                    
                    # Ensure they are lists of lists with correct shapes
                    # U: (n*m, R) -> transpose
                    U_list = np.array(U_list).T.tolist()
                    V_list = np.array(V_list).T.tolist()
                    W_list = np.array(W_list).T.tolist()
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    # If snapping fails, maybe the real solution is valid but not integer?
                    # The task says "integer/half-integer entries".
                    # So we must snap. If snapping fails, this restart is a failure.
                    pass
        
        return None

    # Try ranks from 40 to 44
    # We know 44 works.
    # Let's try 40, 41, 42, 43, 44.
    # If we find a lower rank, we return it.
    # If not, we return the known 44 solution.
    
    best_result = None
    
    # Fallback: The known rank 44 solution
    def get_fallback():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
        
        for block_idx in range(4):
            k_start = block_idx * 2
            steps = [
                {'U': {0*m + 1: 1, 1*m + 2: 1}, 'V': {1*p + k_start: 1, 2*p + (k_start+1): 1}, 'W': {k_start*n + 0: 1, (k_start+1)*n + 1: 1}},
                {'U': {1*m + 1: 1, 1*m + 2: 1}, 'V': {1*p + k_start: 1}, 'W': {k_start*n + 1: 1, (k_start+1)*n + 1: -1}},
                {'U': {0*m + 1: 1}, 'V': {1*p + (k_start+1): 1, 2*p + (k_start+1): -1}, 'W': {(k_start+1)*n + 0: 1, (k_start+1)*n + 1: 1}},
                {'U': {1*m + 2: 1}, 'V': {2*p + k_start: 1, 1*p + k_start: -1}, 'W': {k_start*n + 0: 1, k_start*n + 1: 1}},
                {'U': {0*m + 1: 1, 0*m + 2: 1}, 'V': {2*p + (k_start+1): 1}, 'W': {k_start*n + 0: -1, (k_start+1)*n + 0: 1}},
                {'U': {1*m + 1: 1, 0*m + 1: -1}, 'V': {1*p + k_start: 1, 1*p + (k_start+1): 1}, 'W': {(k_start+1)*n + 1: 1}},
                {'U': {0*m + 2: 1, 1*m + 2: -1}, 'V': {2*p + k_start: 1, 2*p + (k_start+1): 1}, 'W': {k_start*n + 0: 1}}
            ]
            for step in steps:
                for u_idx, val in step['U'].items():
                    U[u_idx][col] += val
                for v_idx, val in step['V'].items():
                    V[v_idx][col] += val
                for w_idx, val in step['W'].items():
                    W[w_idx][col] += val
                col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find better rank
    for R in range(40, 45):
        if time.time() - start_time > timeout:
            break
        result = try_rank(R, max_restarts=10)
        if result is not None:
            best_result = result
            break # Found lower rank, stop
            
    if best_result is None:
        best_result = get_fallback()
        
    return best_result
# EVOLVE-BLOCK-END
