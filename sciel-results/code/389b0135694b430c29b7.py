"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

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
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if not np.isclose(T[u, v, w], 1.0, atol=tol):
                    return False
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
    deadline = start_time + 140
    
    # Fallback: Schoolbook (Rank 36)
    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_result = _to_lists(U0, V0, W0)
    best_rank = n * m * p
    
    # Precompute valid/invalid index arrays
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

    valid_arr = np.array(valid_indices, dtype=int)
    invalid_arr = np.array(invalid_indices, dtype=int)
    u_v, v_v, w_v = valid_arr[:, 0], valid_arr[:, 1], valid_arr[:, 2]
    u_i, v_i, w_i = invalid_arr[:, 0], invalid_arr[:, 1], invalid_arr[:, 2]
    
    # Build full tensor target
    target_T = np.zeros((n*m, m*p, p*n))
    for (u, v, w) in valid_indices:
        target_T[u, v, w] = 1.0

    def loss_and_grad(params, U_shape, V_shape, W_shape):
        r = U_shape[1]
        num_u = U_shape[0] * r
        num_v = V_shape[0] * r
        num_w = W_shape[0] * r
        
        U_flat = params[:num_u]
        V_flat = params[num_u:num_u+num_v]
        W_flat = params[num_u+num_v:num_u+num_v+num_w]
        
        U = U_flat.reshape(U_shape)
        V = V_flat.reshape(V_shape)
        W = W_flat.reshape(W_shape)
        
        # Compute T_approx = U V^T W^T (tensor product)
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Error tensor
        err = T_approx - target_T
        loss = np.sum(err ** 2)
        
        # Gradients via einsum
        # dLoss/dU[u,r] = 2 * sum_{v,w} err[u,v,w] * V[v,r] * W[w,r]
        # dLoss/dV[v,r] = 2 * sum_{u,w} err[u,v,w] * U[u,r] * W[w,r]
        # dLoss/dW[w,r] = 2 * sum_{u,v} err[u,v,w] * U[u,r] * V[v,r]
        
        # Precompute intermediate contractions
        # For U: err contracted with V and W
        # err @ V @ W along appropriate axes
        # dU[u,r] = 2 * sum_{v,w} err[u,v,w] * V[v,r] * W[w,r]
        # = 2 * sum_w (sum_v err[u,v,w] * V[v,r]) * W[w,r]
        # Let's do: temp1[u,v,r] = sum_w err[u,v,w] * W[w,r] -> einsum 'uvw,wr->uvr'
        # Then dU[u,r] = 2 * sum_v temp1[u,v,r] * V[v,r] -> einsum 'uvr,vr->ur'
        
        temp1 = np.einsum('uvw,wr->uvr', err, W)  # (nm, mp, r)
        grad_U = 2.0 * np.einsum('uvr,vr->ur', temp1, V)  # (nm, r)
        
        # dV[v,r] = 2 * sum_{u,w} err[u,v,w] * U[u,r] * W[w,r]
        # temp2[u,v,r] = sum_w err[u,v,w] * W[w,r] (same as temp1)
        # dV[v,r] = 2 * sum_u temp2[u,v,r] * U[u,r]
        grad_V = 2.0 * np.einsum('uvr,ur->vr', temp1, U)  # (mp, r)
        
        # dW[w,r] = 2 * sum_{u,v} err[u,v,w] * U[u,r] * V[v,r]
        # temp3[u,v,r] = sum_w err[u,v,w] * W[w,r] (same as temp1)
        # dW[w,r] = 2 * sum_{u,v} err[u,v,w] * U[u,r] * V[v,r]
        # Let's do: temp4[u,w,r] = sum_v err[u,v,w] * V[v,r] -> einsum 'uvw,vr->uw r'
        temp4 = np.einsum('uvw,vr->uwr', err, V)  # (nm, pn, r)
        grad_W = 2.0 * np.einsum('uwr,ur->wr', temp4, U)  # (pn, r)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def snap_and_check(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if _verify(U_s, V_s, W_s, n, m, p, tol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None, None

    def make_schoolbook_init(r):
        """Create a schoolbook-based initialization with r columns."""
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        # Fill with schoolbook pattern, cycling through
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= r:
                        break
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        # Add small random perturbation to remaining columns
        for c in range(col, r):
            U[:, c] = np.random.randn(n * m) * 0.1
            V[:, c] = np.random.randn(m * p) * 0.1
            W[:, c] = np.random.randn(p * n) * 0.1
        return U, V, W

    # Search for lower rank
    from scipy.optimize import minimize
    
    # Try ranks from 25 to best_rank-1
    for target_rank in range(25, best_rank):
        if time.time() > deadline - 15:
            break
            
        U_shape = (n * m, target_rank)
        V_shape = (m * p, target_rank)
        W_shape = (p * n, target_rank)
        
        found = False
        num_restarts = 15 if target_rank <= 30 else 8
        
        for attempt in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with schoolbook + perturbation
            U, V, W = make_schoolbook_init(target_rank)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(loss_and_grad, params, 
                               args=(U_shape, V_shape, W_shape), 
                               jac=True, method='L-BFGS-B', 
                               options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            except Exception:
                continue
                
            if res.fun < 1e-10:
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
