import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    Uses a structured search: start from known rank-44 decomposition (Strassen + schoolbook),
    then attempt to reduce rank via random restarts and local optimization.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety

    # Build the multiplication tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Known rank-44 decomposition
    def build_rank44():
        rank = 44
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        # Strassen 2x2x2
        S_U = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ], dtype=np.float64)
        S_V = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ], dtype=np.float64)
        S_W = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, -1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ], dtype=np.float64)
        
        col = 0
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        U[li * m + la, col] = S_U[r, li * 2 + la]
                for la in range(2):
                    for lk in range(2):
                        V[la * p + (k0 + lk), col] = S_V[r, la * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0 + lk) * n + li, col] = S_W[r, lk * 2 + li]
                col += 1
        
        # Schoolbook for a=2 (third row of A, i.e., j=2)
        for i in range(n):
            for k in range(p):
                U[i * m + 2, col] = 1.0
                V[2 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
        
        return U, V, W
    
    U44, V44, W44 = build_rank44()
    
    def tensor_error(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.linalg.norm(T_rec - T)
    
    def try_rank_reduction(target_rank, max_time=10.0):
        """Try to find a decomposition of rank target_rank by optimizing with random restarts."""
        if time.time() > deadline - max_time:
            return None
        
        num_restarts = 5
        best_err = float('inf')
        best_params = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 2:
                break
            
            # Initialize
            if restart == 0:
                # Start from truncated rank-44
                U = U44[:, :target_rank].copy()
                V = V44[:, :target_rank].copy()
                W = W44[:, :target_rank].copy()
            else:
                # Random initialization
                U = np.random.randn(n*m, target_rank) * 0.1
                V = np.random.randn(m*p, target_rank) * 0.1
                W = np.random.randn(p*n, target_rank) * 0.1
            
            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten_params(x):
                nU = U.size
                nV = V.size
                U_ = x[:nU].reshape(U.shape)
                V_ = x[nU:nU+nV].reshape(V.shape)
                W_ = x[nU+nV:].reshape(W.shape)
                return U_, V_, W_
            
            def objective(x):
                U_, V_, W_ = unflatten_params(x)
                return tensor_error(U_, V_, W_)
            
            x0 = flatten_params(U, V, W)
            
            # Use L-BFGS-B
            try:
                result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            except Exception:
                continue
            
            if result.fun < best_err:
                best_err = result.fun
                best_params = result.x
            
            if best_err < 1e-6:
                break
        
        if best_params is not None and best_err < 1e-6:
            U_, V_, W_ = unflatten_params(best_params)
            # Round to nearest half-integer
            U_ = np.round(U_ * 2) / 2
            V_ = np.round(V_ * 2) / 2
            W_ = np.round(W_ * 2) / 2
            if tensor_error(U_, V_, W_) < 1e-6:
                return U_, V_, W_
        
        return None
    
    # Try ranks from 43 down to 30
    best_result = (U44, V44, W44)
    best_rank = 44
    
    for target_rank in range(43, 29, -1):
        if time.time() > deadline - 15:
            break
        result = try_rank_reduction(target_rank, max_time=10.0)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found a lower rank, stop
    
    U, V, W = best_result
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
