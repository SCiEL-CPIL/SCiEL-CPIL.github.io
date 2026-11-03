import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def schoolbook():
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

    def check_validity(U, V, W, tol=1e-8):
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(U.shape[1]):
                            s += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def objective_and_grad(params, r):
        # Unpack
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r :].reshape(p*n, r)
        
        # Compute T_est
        # T_est[u, v, w] = sum_k U[u, k] V[v, k] W[w, k]
        T_est = np.einsum('uk,vk,wk->uvw', U, V, W)
        diff = T - T_est
        loss = np.sum(diff**2)
        
        # Gradients
        # dL/dU[u, k] = 2 * sum_{v, w} diff[u, v, w] V[v, k] W[w, k]
        grad_U = 2 * np.einsum('uvw,vk,wk->uk', diff, V, W)
        grad_V = 2 * np.einsum('uvw,uk,wk->vk', diff, U, W)
        grad_W = 2 * np.einsum('uvw,uk,vk->wk', diff, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad

    def try_optimize(r, num_restarts=5, seed=42):
        rng = np.random.RandomState(seed)
        best_result = None
        best_loss = np.inf
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            init_U = rng.randn(n*m, r) * 0.1
            init_V = rng.randn(m*p, r) * 0.1
            init_W = rng.randn(p*n, r) * 0.1
            
            x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            res = minimize(
                lambda x: objective_and_grad(x, r)[0],
                x0,
                jac=lambda x: objective_and_grad(x, r)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 5:
                return None
                
            if res.success or res.fun < best_loss:
                best_loss = res.fun
                if best_loss < 1e-6:
                    # Extract U, V, W
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    # Round and verify
                    Ur = round_to_half(U)
                    Vr = round_to_half(V)
                    Wr = round_to_half(W)
                    
                    T_est_r = np.einsum('uk,vk,wk->uvw', Ur, Vr, Wr)
                    err_r = np.linalg.norm(T - T_est_r)
                    
                    if err_r < 1e-8:
                        return Ur, Vr, Wr, r
                    # If rounding fails, we might still return the float if we are desperate?
                    # No, must be half-integers.
                    # Store best float result in case we need to fallback?
                    # But we need valid half-integers.
                    best_result = (Ur, Vr, Wr, r) # Keep rounded version even if not perfect?
                    # If we return non-perfect, score is 0.
                    # So we only return if perfect.
        
        return None

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    # Try ranks
    # L-BFGS-B is faster than ALS for this size.
    ranks_to_try = [143, 140, 130, 120, 100, 80, 60, 50, 40, 36]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        result = try_optimize(r, num_restarts=3, seed=42 + r)
        if result is not None:
            U, V, W, rank = result
            if check_validity(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
