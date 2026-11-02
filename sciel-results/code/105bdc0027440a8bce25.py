import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor M of shape (dim_u, dim_v, dim_w)
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
                
    def loss_and_grad(params, rank):
        u_flat = params[:dim_u * rank]
        v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
        w_flat = params[dim_u * rank + dim_v * rank : ]
        
        U = u_flat.reshape(dim_u, rank)
        V = v_flat.reshape(dim_v, rank)
        W = w_flat.reshape(dim_w, rank)
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - target
        loss = np.sum(diff * diff)
        
        grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
        return loss, grad

    start_time = time.time()
    deadline = 147.0  # 3 seconds margin before 150s kill
    
    best_result = None
    
    # Fallback: Schoolbook
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
        return U, V, W, rank

    def verify_solution(U, V, W):
        """Verify that U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        # Check exact match: all entries should be exactly 0 or 1
        # Since we snap to half-integers, the result should be exact
        max_diff = np.max(np.abs(T - target))
        return max_diff < 1e-12

    # Try ranks from 50 down to 25
    for target_rank in range(50, 24, -1):
        if time.time() > deadline:
            break
            
        found = False
        # More restarts for lower ranks
        n_restarts = 20 if target_rank <= 40 else 15
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Use different initialization strategies
            np.random.seed(12345 + target_rank * 100 + restart)
            
            # Strategy 1: Small random perturbation around zero
            scale = 0.1
            u0 = np.random.randn(dim_u * target_rank) * scale
            v0 = np.random.randn(dim_v * target_rank) * scale
            w0 = np.random.randn(dim_w * target_rank) * scale
            
            params = np.concatenate([u0, v0, w0])
            
            def fun(params):
                loss, grad = loss_and_grad(params, target_rank)
                return loss, grad
                
            try:
                res = minimize(fun, params, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                
                # Check if we got a good solution
                if res.fun < 1e-10:
                    u_final = res.x[:dim_u * target_rank].reshape(dim_u, target_rank)
                    v_final = res.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                    w_final = res.x[dim_u * target_rank + dim_v * target_rank : ].reshape(dim_w, target_rank)
                    
                    # Snap to nearest 0.5
                    u_snap = np.round(u_final * 2.0) / 2.0
                    v_snap = np.round(v_final * 2.0) / 2.0
                    w_snap = np.round(w_final * 2.0) / 2.0
                    
                    # Verify snapped solution
                    if verify_solution(u_snap, v_snap, w_snap):
                        found = True
                        best_result = (u_snap, v_snap, w_snap, target_rank)
                        break
                        
            except Exception:
                continue

        if found:
            break

    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W, rank = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
