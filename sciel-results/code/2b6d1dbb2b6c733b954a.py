import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    start_time = time.time()
    deadline = 140.0
    
    # Build target tensor
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0

    def verify_solution(U, V, W):
        """Verify exact decomposition with half-integer entries."""
        # Check entries are half-integers
        for M in [U, V, W]:
            if not np.allclose(M * 2, np.round(M * 2), atol=1e-9):
                return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        max_diff = np.max(np.abs(T - target))
        return max_diff < 1e-9

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def try_rank(r, max_restarts=20):
        """Try to find a rank-r decomposition."""
        start = time.time()
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            if time.time() - start > 5.0: # Time limit per rank
                break
                
            np.random.seed(42 + r * 100 + restart)
            
            # Initialize
            U = np.zeros((dim_u, r))
            V = np.zeros((dim_v, r))
            W = np.zeros((dim_w, r))
            
            # Strategy: Random sparse initialization with small integer values
            # Ensure some density
            for col in range(r):
                # Pick random non-zeros
                # U: pick 1-2 indices
                u_idx = np.random.choice(dim_u, size=np.random.randint(1, 3), replace=False)
                for idx in u_idx:
                    U[idx, col] = np.random.choice([-1, 1])
                    
                v_idx = np.random.choice(dim_v, size=np.random.randint(1, 3), replace=False)
                for idx in v_idx:
                    V[idx, col] = np.random.choice([-1, 1])
                    
                w_idx = np.random.choice(dim_w, size=np.random.randint(1, 3), replace=False)
                for idx in w_idx:
                    W[idx, col] = np.random.choice([-1, 1])
            
            # Flatten params
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def loss_fn(params):
                U_ = params[:dim_u*r].reshape(dim_u, r)
                V_ = params[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
                W_ = params[dim_u*r+dim_v*r:].reshape(dim_w, r)
                T = np.einsum('ar,br,cr->abc', U_, V_, W_)
                return np.sum((T - target)**2)
            
            try:
                res = minimize(loss_fn, params, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-12})
                if res.fun < 1e-8:
                    U_f = res.x[:dim_u*r].reshape(dim_u, r)
                    V_f = res.x[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
                    W_f = res.x[dim_u*r+dim_v*r:].reshape(dim_w, r)
                    
                    # Snap to half-integers
                    U_s = np.round(U_f * 2) / 2
                    V_s = np.round(V_f * 2) / 2
                    W_s = np.round(W_f * 2) / 2
                    
                    if verify_solution(U_s, V_s, W_s):
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            except Exception:
                pass
        return None

    # Try ranks from 45 to 49
    # Rank 45 is the theoretical lower bound estimate 2n^2 - n
    for r in [45, 46, 47, 48, 49]:
        if time.time() > deadline:
            break
        result = try_rank(r, max_restarts=10)
        if result is not None:
            return result

    # Fallback to schoolbook
    return get_schoolbook()
