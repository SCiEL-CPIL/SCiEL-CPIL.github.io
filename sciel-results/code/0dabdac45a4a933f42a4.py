"""Rewritten program for Matrix multiplication 3,3,4 rank-minimal decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _to_lists(U, V, W):
    def clean_matrix(M):
        # Ensure entries are exactly representable as half-integers
        # The task allows integer/half-integer entries
        M_clean = np.round(M * 2) / 2
        return M_clean.tolist()
    
    return {
        "U": clean_matrix(U),
        "V": clean_matrix(V),
        "W": clean_matrix(W)
    }

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    
    # Compute full tensor T = sum_r U[:,r] V[:,r] W[:,r]
    # T[u, v, w]
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    # Check valid indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if not np.isclose(T[u, v, w], 1.0, atol=tol):
                    return False
    
    # Check invalid indices
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_set.add((u, v, w))
                
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) not in valid_set:
                    if not np.isclose(T[u, v, w], 0.0, atol=tol):
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precompute valid and invalid indices
    valid_indices = []
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_indices.append((u, v, w))
                valid_set.add((u, v, w))
                
    invalid_indices = []
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) not in valid_set:
                    invalid_indices.append((u, v, w))
    
    valid_u = np.array([u for u, v, w in valid_indices])
    valid_v = np.array([v for u, v, w in valid_indices])
    valid_w = np.array([w for u, v, w in valid_indices])
    
    invalid_u = np.array([u for u, v, w in invalid_indices])
    invalid_v = np.array([v for u, v, w in invalid_indices])
    invalid_w = np.array([w for u, v, w in invalid_indices])
    
    # Target Tensor
    Target = np.zeros((n*m, m*p, p*n))
    for (u, v, w) in valid_indices:
        Target[u, v, w] = 1.0

    def loss_and_grad(params, U_shape, V_shape, W_shape, r):
        num_u = U_shape[0] * r
        num_v = V_shape[0] * r
        num_w = W_shape[0] * r
        
        U_flat = params[:num_u]
        V_flat = params[num_u:num_u+num_v]
        W_flat = params[num_u+num_v:num_u+num_v+num_w]
        
        U = U_flat.reshape(U_shape)
        V = V_flat.reshape(V_shape)
        W = W_flat.reshape(W_shape)
        
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        T_err = T_approx - Target
        
        loss = np.sum(T_err**2)
        
        grad_U = 2.0 * np.einsum('uvw,vr,wr->ur', T_err, V, W)
        grad_V = 2.0 * np.einsum('uvw,ur,wr->vr', T_err, U, W)
        grad_W = 2.0 * np.einsum('uvw,ur,vr->wr', T_err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def snap_and_check(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        if _verify(U_s, V_s, W_s, n, m, p, tol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None, None

    best_result = None
    best_rank = n * m * p
    
    # Initialize with schoolbook as fallback
    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_result = _to_lists(U0, V0, W0)
    
    # Search for lower rank
    # Known optimal rank for 3x3x4 is 19.
    for target_rank in range(19, best_rank + 1):
        if time.time() > deadline - 5:
            break
            
        U_shape = (n * m, target_rank)
        V_shape = (m * p, target_rank)
        W_shape = (p * n, target_rank)
        
        found = False
        
        # Increase restarts and use different initialization strategies
        num_restarts = 100
        
        for attempt in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Strategy 1: Sparse Random Integer Initialization
            # Many optimal decompositions are sparse with small integer entries.
            U = np.zeros(U_shape)
            V = np.zeros(V_shape)
            W = np.zeros(W_shape)
            
            # Add random non-zero entries
            # Density ~ 1/3
            mask_u = np.random.rand(*U_shape) < 0.3
            mask_v = np.random.rand(*V_shape) < 0.3
            mask_w = np.random.rand(*W_shape) < 0.3
            
            U[mask_u] = np.random.choice([-1, 1], size=np.sum(mask_u))
            V[mask_v] = np.random.choice([-1, 1], size=np.sum(mask_v))
            W[mask_w] = np.random.choice([-1, 1], size=np.sum(mask_w))
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use L-BFGS-B with tight tolerance
                res = minimize(loss_and_grad, params, 
                               args=(U_shape, V_shape, W_shape, target_rank), 
                               jac=True, method='L-BFGS-B', 
                               options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-8, 'gtol': 1e-5})
            except Exception:
                continue
                
            if res.fun < 1e-4:
                num_u = U_shape[0] * target_rank
                num_v = V_shape[0] * target_rank
                num_w = W_shape[0] * target_rank
                U_final = res.x[:num_u].reshape(U_shape)
                V_final = res.x[num_u:num_u+num_v].reshape(V_shape)
                W_final = res.x[num_u+num_v:num_u+num_v+num_w].reshape(W_shape)
                
                success, U_s, V_s, W_s = snap_and_check(U_final, V_final, W_final)
                if success:
                    best_rank = target_rank
                    best_result = _to_lists(U_s, V_s, W_s)
                    found = True
                    break
        
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
