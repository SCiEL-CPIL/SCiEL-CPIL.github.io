import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
                
    def compute_tensor(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)

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

    def local_search_half_integer(U, V, W, max_iters=200):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for _ in range(max_iters):
            T = np.einsum('ar,br,cr->abc', U, V, W)
            E = T - target
            
            grad_u = np.einsum('abc,br,cr->ar', E, V, W)
            norm_r_u = np.einsum('br,cr->r', V**2, W**2)
            
            grad_v = np.einsum('abc,ar,cr->br', E, U, W)
            norm_r_v = np.einsum('ar,cr->r', U**2, W**2)
            
            grad_w = np.einsum('abc,ar,br->cr', E, U, V)
            norm_r_w = np.einsum('ar,br->r', U**2, V**2)
            
            best_delta = 0
            best_move = None
            
            dL_u_pos = 0.5 * grad_u + 0.25 * norm_r_u[None, :]
            dL_u_neg = -0.5 * grad_u + 0.25 * norm_r_u[None, :]
            min_dL_u = np.min(np.minimum(dL_u_pos, dL_u_neg))
            if min_dL_u < best_delta:
                flat_min = np.argmin(np.minimum(dL_u_pos, dL_u_neg))
                i, r = np.unravel_index(flat_min, grad_u.shape)
                if dL_u_pos[i, r] < dL_u_neg[i, r]:
                    best_move = ('U', i, r, 0.5)
                else:
                    best_move = ('U', i, r, -0.5)
                best_delta = min_dL_u

            dL_v_pos = 0.5 * grad_v + 0.25 * norm_r_v[None, :]
            dL_v_neg = -0.5 * grad_v + 0.25 * norm_r_v[None, :]
            min_dL_v = np.min(np.minimum(dL_v_pos, dL_v_neg))
            if min_dL_v < best_delta:
                flat_min = np.argmin(np.minimum(dL_v_pos, dL_v_neg))
                i, r = np.unravel_index(flat_min, grad_v.shape)
                if dL_v_pos[i, r] < dL_v_neg[i, r]:
                    best_move = ('V', i, r, 0.5)
                else:
                    best_move = ('V', i, r, -0.5)
                best_delta = min_dL_v

            dL_w_pos = 0.5 * grad_w + 0.25 * norm_r_w[None, :]
            dL_w_neg = -0.5 * grad_w + 0.25 * norm_r_w[None, :]
            min_dL_w = np.min(np.minimum(dL_w_pos, dL_w_neg))
            if min_dL_w < best_delta:
                flat_min = np.argmin(np.minimum(dL_w_pos, dL_w_neg))
                i, r = np.unravel_index(flat_min, grad_w.shape)
                if dL_w_pos[i, r] < dL_w_neg[i, r]:
                    best_move = ('W', i, r, 0.5)
                else:
                    best_move = ('W', i, r, -0.5)
                best_delta = min_dL_w

            if best_move is None:
                break
            
            m_type, i, r, delta = best_move
            if m_type == 'U':
                U[i, r] += delta
            elif m_type == 'V':
                V[i, r] += delta
            else:
                W[i, r] += delta
                
        return U, V, W

    start_time = time.time()
    deadline = 145.0
    
    best_result = None
    
    for target_rank in range(49, 30, -1):
        if time.time() > deadline:
            break
            
        found = False
        num_restarts = 5
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(1000 + target_rank + restart)
            
            if restart % 2 == 0:
                u0 = np.random.randn(dim_u * target_rank) * 0.1
                v0 = np.random.randn(dim_v * target_rank) * 0.1
                w0 = np.random.randn(dim_w * target_rank) * 0.1
            else:
                u0 = np.random.randn(dim_u * target_rank)
                v0 = np.random.randn(dim_v * target_rank)
                w0 = np.random.randn(dim_w * target_rank)
                mask_u = np.random.rand(dim_u, target_rank) > 0.5
                mask_v = np.random.rand(dim_v, target_rank) > 0.5
                mask_w = np.random.rand(dim_w, target_rank) > 0.5
                u0 = u0.reshape(dim_u, target_rank)
                v0 = v0.reshape(dim_v, target_rank)
                w0 = w0.reshape(dim_w, target_rank)
                u0[mask_u] = 0
                v0[mask_v] = 0
                w0[mask_w] = 0
                u0 = u0.flatten()
                v0 = v0.flatten()
                w0 = w0.flatten()
            
            params = np.concatenate([u0, v0, w0])
            
            def fun(params):
                loss, grad = loss_and_grad(params, target_rank)
                return loss, grad
                
            try:
                res = minimize(fun, params, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6})
                
                u_final = res.x[:dim_u * target_rank].reshape(dim_u, target_rank)
                v_final = res.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                w_final = res.x[dim_u * target_rank + dim_v * target_rank : ].reshape(dim_w, target_rank)
                
                u_snap = np.round(u_final * 2.0) / 2.0
                v_snap = np.round(v_final * 2.0) / 2.0
                w_snap = np.round(w_final * 2.0) / 2.0
                
                u_ls, v_ls, w_ls = local_search_half_integer(u_snap, v_snap, w_snap)
                
                T_final = np.einsum('ar,br,cr->abc', u_ls, v_ls, w_ls)
                if np.allclose(T_final, target, atol=1e-9):
                    found = True
                    best_result = (u_ls, v_ls, w_ls, target_rank)
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
