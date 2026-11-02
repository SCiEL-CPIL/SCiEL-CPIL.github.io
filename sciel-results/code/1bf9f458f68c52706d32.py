import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0

    def get_schoolbook():
        """Return schoolbook decomposition as dict of lists."""
        rank = n * m * p  # 50
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
    
    def verify_decomposition(U, V, W, atol=1e-9):
        """Verify that U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=atol)
    
    def snap_and_verify(U, V, W):
        """Snap to half-integers and verify. Returns (U_snap, V_snap, W_snap) or None."""
        u_snap = np.round(U * 2.0) / 2.0
        v_snap = np.round(V * 2.0) / 2.0
        w_snap = np.round(W * 2.0) / 2.0
        
        if verify_decomposition(u_snap, v_snap, w_snap):
            return (u_snap, v_snap, w_snap)
        
        # Try cleaning small values
        u_clean = np.where(np.abs(u_snap) < 1e-6, 0, u_snap)
        v_clean = np.where(np.abs(v_snap) < 1e-6, 0, v_snap)
        w_clean = np.where(np.abs(w_snap) < 1e-6, 0, w_snap)
        if verify_decomposition(u_clean, v_clean, w_clean):
            return (u_clean, v_clean, w_clean)
        
        # Try snapping to integers only
        u_int = np.round(U)
        v_int = np.round(V)
        w_int = np.round(W)
        if verify_decomposition(u_int, v_int, w_int):
            return (u_int, v_int, w_int)
        
        return None

    def try_rank_als(rank, max_iters=200, tol=1e-10):
        """Try to find a decomposition using ALS."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + rank)
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        # Normalize to prevent divergence
        for r in range(rank):
            norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            if norm > 0:
                scale = (1.0 / norm) ** (1.0/3.0)
                U[:, r] *= scale
                V[:, r] *= scale
                W[:, r] *= scale
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            T_u = np.einsum('abc,br,cr->ar', target, V, W)
            G_u = (V.T @ V) * (W.T @ W)
            G_u += 1e-8 * np.eye(rank)
            U_new = np.linalg.solve(G_u, T_u.T).T
            
            # Update V
            T_v = np.einsum('abc,ar,cr->br', target, U_new, W)
            G_v = (U_new.T @ U_new) * (W.T @ W)
            G_v += 1e-8 * np.eye(rank)
            V_new = np.linalg.solve(G_v, T_v.T).T
            
            # Update W
            T_w = np.einsum('abc,ar,br->cr', target, U_new, V_new)
            G_w = (U_new.T @ U_new) * (V_new.T @ V_new)
            G_w += 1e-8 * np.eye(rank)
            W_new = np.linalg.solve(G_w, T_w.T).T
            
            # Check convergence
            err = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if err < tol:
                break
        
        result = snap_and_verify(U, V, W)
        return result

    def loss_and_grad(params, rank):
        """Compute loss and gradient for L-BFGS-B."""
        u_flat = params[:dim_u * rank]
        v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
        w_flat = params[dim_u * rank + dim_v * rank : ]
        
        U = u_flat.reshape(dim_u, rank)
        V = v_flat.reshape(dim_v, rank)
        W = w_flat.reshape(dim_w, rank)
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - target
        loss = np.sum(diff * diff)
        
        # Gradient
        grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
        return loss, grad

    def try_rank_lbfgs(rank, max_restarts=5):
        """Try to find a decomposition using L-BFGS-B."""
        if time.time() > deadline:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(1000 + rank * 10 + restart)
            u0 = np.random.randn(dim_u * rank) * 0.1
            v0 = np.random.randn(dim_v * rank) * 0.1
            w0 = np.random.randn(dim_w * rank) * 0.1
            
            params = np.concatenate([u0, v0, w0])
            
            def fun(params):
                return loss_and_grad(params, rank)
            
            try:
                res = minimize(fun, params, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    u_final = res.x[:dim_u * rank].reshape(dim_u, rank)
                    v_final = res.x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    w_final = res.x[dim_u * rank + dim_v * rank : ].reshape(dim_w, rank)
                    
                    result = snap_and_verify(u_final, v_final, w_final)
                    if result is not None:
                        return result
            except Exception:
                continue
        
        return None

    best_result = None
    
    # Try ranks from low to high, using both ALS and L-BFGS-B
    for target_rank in [25, 30, 32, 35, 40, 45]:
        if time.time() > deadline:
            break
        
        # Try ALS first (faster per iteration)
        result = try_rank_als(target_rank, max_iters=150)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
        
        # Try L-BFGS-B if ALS didn't work
        if time.time() < deadline:
            result = try_rank_lbfgs(target_rank, max_restarts=3)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
