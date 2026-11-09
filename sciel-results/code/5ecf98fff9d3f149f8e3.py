import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    T_flat = T.flatten()
    
    def check_decomposition(U, V, W, atol=1e-8):
        """Check if U, V, W exactly decompose T with half-integer entries."""
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        
        if not (np.allclose(U, U_h, atol=0.01) and np.allclose(V, V_h, atol=0.01) and np.allclose(W, W_h, atol=0.01)):
            return None
            
        T_approx = np.einsum('ic,jc,kc->ijk', U_h, V_h, W_h)
        if np.allclose(T_approx, T, atol=atol):
            return {
                "U": U_h.tolist(),
                "V": V_h.tolist(),
                "W": W_h.tolist()
            }
        return None
    
    def try_rank_als(r, max_iters=80, num_restarts=50, deadline=None):
        """Try ALS with structured initialization."""
        best_result = None
        
        for restart in range(num_restarts):
            if deadline and time.time() > deadline:
                break
                
            np.random.seed(999 + restart * 13)
            
            # Structured initialization: use small random values
            scale = 0.3
            U = np.random.randn(dim_u, r) * scale
            V = np.random.randn(dim_v, r) * scale
            W = np.random.randn(dim_w, r) * scale
            
            # Also try a structured init based on tensor sparsity pattern
            if restart % 3 == 0:
                # Initialize based on the non-zero pattern
                U = np.zeros((dim_u, r))
                V = np.zeros((dim_v, r))
                W = np.zeros((dim_w, r))
                for idx in range(r):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    U[I, idx] = 1.0
                    V[J, idx] = 1.0
                    W[K, idx] = 1.0
                # Add small random perturbation to break symmetry
                U += np.random.randn(dim_u, r) * 0.05
                V += np.random.randn(dim_v, r) * 0.05
                W += np.random.randn(dim_w, r) * 0.05
            
            for it in range(max_iters):
                if deadline and time.time() > deadline:
                    return best_result
                
                # Update U: for each row I, solve least squares
                # T[I, J, K] = sum_c U[I, c] * V[J, c] * W[K, c]
                # For fixed I: b[J*K] = T[I, J, K], A[(J*K), c] = V[J, c]*W[K, c]
                U_new = np.zeros_like(U)
                for I in range(dim_u):
                    b = T[I, :, :].flatten()
                    A = (V * W[:, None, :]).reshape(dim_v * dim_w, r)  # V[j,c]*W[k,c]
                    try:
                        U_new[I, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        U_new[I, :] = U[I, :]
                
                # Update V
                V_new = np.zeros_like(V)
                for J in range(dim_v):
                    b = T[:, J, :].flatten()
                    A = (U_new * W[:, None, :]).reshape(dim_u * dim_w, r)
                    try:
                        V_new[J, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        V_new[J, :] = V[J, :]
                
                # Update W
                W_new = np.zeros_like(W)
                for K in range(dim_w):
                    b = T[:, :, K].flatten()
                    A = (U_new * V_new[:, None, :]).reshape(dim_u * dim_v, r)
                    try:
                        W_new[K, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        W_new[K, :] = W[K, :]
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence and exactness periodically
                if it % 4 == 0:
                    res = check_decomposition(U, V, W)
                    if res:
                        return res
            
            # Final check
            res = check_decomposition(U, V, W)
            if res:
                return res
        
        return None
    
    def try_rank_local_search(r, num_starts=20, deadline=None):
        """Try local search from multiple starting points."""
        best_result = None
        
        for start in range(num_starts):
            if deadline and time.time() > deadline:
                break
                
            np.random.seed(555 + start * 7)
            
            # Start from structured initialization
            U = np.random.randn(dim_u, r) * 0.5
            V = np.random.randn(dim_v, r) * 0.5
            W = np.random.randn(dim_w, r) * 0.5
            
            # Run a few ALS iterations to get close to a solution
            for _ in range(30):
                U_new = np.zeros_like(U)
                for I in range(dim_u):
                    b = T[I, :, :].flatten()
                    A = (V * W[:, None, :]).reshape(dim_v * dim_w, r)
                    try:
                        U_new[I, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        U_new[I, :] = U[I, :]
                
                V_new = np.zeros_like(V)
                for J in range(dim_v):
                    b = T[:, J, :].flatten()
                    A = (U_new * W[:, None, :]).reshape(dim_u * dim_w, r)
                    try:
                        V_new[J, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        V_new[J, :] = V[J, :]
                
                W_new = np.zeros_like(W)
                for K in range(dim_w):
                    b = T[:, :, K].flatten()
                    A = (U_new * V_new[:, None, :]).reshape(dim_u * dim_v, r)
                    try:
                        W_new[K, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        W_new[K, :] = W[K, :]
                
                U, V, W = U_new, V_new, W_new
            
            # Local search: perturb and check
            for _ in range(50):
                if deadline and time.time() > deadline:
                    break
                    
                # Small random perturbation
                U_p = U + np.random.randn(dim_u, r) * 0.01
                V_p = V + np.random.randn(dim_v, r) * 0.01
                W_p = W + np.random.randn(dim_w, r) * 0.01
                
                res = check_decomposition(U_p, V_p, W_p)
                if res:
                    return res
                
                # Also try snapping to half-integers and checking
                U_s = np.round(U_p * 2) / 2
                V_s = np.round(V_p * 2) / 2
                W_s = np.round(W_p * 2) / 2
                res = check_decomposition(U_s, V_s, W_s)
                if res:
                    return res
            
            # Check the current point
            res = check_decomposition(U, V, W)
            if res:
                return res
        
        return None
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try sub-schoolbook ranks in order
    for r in [25, 26, 27, 28, 29]:
        if time.time() > deadline:
            break
        
        # Try ALS approach
        res = try_rank_als(r, max_iters=60, num_restarts=30, deadline=deadline)
        if res:
            return res
        
        # Try local search approach
        res = try_rank_local_search(r, num_starts=10, deadline=deadline)
        if res:
            return res
    
    # Fallback to schoolbook
    return schoolbook_decomposition()
