import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid_indices.append((alpha, beta, gamma))
                
    total_indices = []
    for alpha in range(n * m):
        for beta in range(m * p):
            for gamma in range(p * n):
                total_indices.append((alpha, beta, gamma))
                
    valid_set = set(valid_indices)
    invalid_indices = [idx for idx in total_indices if idx not in valid_set]
    
    def compute_loss(U, V, W):
        T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
        loss = 0.0
        for (a, b, g) in valid_indices:
            loss += (T_approx[a, b, g] - 1.0) ** 2
        for (a, b, g) in invalid_indices:
            loss += T_approx[a, b, g] ** 2
        return loss

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
        
        T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (a, b, g) in valid_indices:
            val = T_approx[a, b, g]
            err = val - 1.0
            loss += err * err
            grad_U[a, :] += 2.0 * err * V[b, :] * W[g, :]
            grad_V[b, :] += 2.0 * err * U[a, :] * W[g, :]
            grad_W[g, :] += 2.0 * err * U[a, :] * V[b, :]
            
        for (a, b, g) in invalid_indices:
            val = T_approx[a, b, g]
            err = val - 0.0
            loss += err * err
            grad_U[a, :] += 2.0 * err * V[b, :] * W[g, :]
            grad_V[b, :] += 2.0 * err * U[a, :] * W[g, :]
            grad_W[g, :] += 2.0 * err * U[a, :] * V[b, :]
            
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def snap_and_check(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        T_approx = np.einsum('ac,bc,dc->abd', U_s, V_s, W_s)
        
        for (a, b, g) in valid_indices:
            if not np.isclose(T_approx[a, b, g], 1.0, atol=1e-9):
                return False
        for (a, b, g) in invalid_indices:
            if not np.isclose(T_approx[a, b, g], 0.0, atol=1e-9):
                return False
        return True

    def local_search_halfint(U, V, W, max_iters=200):
        """Local search on half-integer grid to find exact solution"""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        best_loss = compute_loss(U_s, V_s, W_s)
        best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
        
        for it in range(max_iters):
            improved = False
            # Try perturbing one random element by +/- 0.5
            for _ in range(100):
                which = np.random.choice(3)
                if which == 0:
                    idx = np.random.choice(U_s.size)
                    i, j = np.unravel_index(idx, U_s.shape)
                    delta = np.random.choice([-0.5, 0.5])
                    U_try = U_s.copy()
                    U_try[i, j] += delta
                    V_try, W_try = V_s, W_s
                elif which == 1:
                    idx = np.random.choice(V_s.size)
                    i, j = np.unravel_index(idx, V_s.shape)
                    delta = np.random.choice([-0.5, 0.5])
                    V_try = V_s.copy()
                    V_try[i, j] += delta
                    U_try, W_try = U_s, W_s
                else:
                    idx = np.random.choice(W_s.size)
                    i, j = np.unravel_index(idx, W_s.shape)
                    delta = np.random.choice([-0.5, 0.5])
                    W_try = W_s.copy()
                    W_try[i, j] += delta
                    U_try, V_try = U_s, V_s
                
                loss_try = compute_loss(U_try, V_try, W_try)
                
                if loss_try < best_loss - 1e-12:
                    best_loss = loss_try
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    U_s, V_s, W_s = U_try, V_try, W_try
                    improved = True
                    break
            
            if not improved:
                break
        
        return best_U, best_V, best_W, best_loss

    def get_schoolbook():
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

    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Try ranks from 25 to 35
    for r in range(25, 36):
        if time.time() > deadline:
            break
            
        U_shape = (n * m, r)
        V_shape = (m * p, r)
        W_shape = (p * n, r)
        
        found = False
        time_left = deadline - time.time()
        num_restarts = max(5, min(25, int(time_left / 2)))
        
        for attempt in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with Gaussian
            U = np.random.randn(*U_shape) * 0.5
            V = np.random.randn(*V_shape) * 0.5
            W = np.random.randn(*W_shape) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(loss_and_grad, params, args=(U_shape, V_shape, W_shape, r), 
                               jac=True, method='L-BFGS-B', 
                               options={'maxiter': 300, 'maxfun': 3000})
            except Exception:
                continue
                
            if res.fun < 1e-3:
                num_u = U_shape[0] * r
                num_v = V_shape[0] * r
                num_w = W_shape[0] * r
                U_final = res.x[:num_u].reshape(U_shape)
                V_final = res.x[num_u:num_u+num_v].reshape(V_shape)
                W_final = res.x[num_u+num_v:num_u+num_v+num_w].reshape(W_shape)
                
                # Try direct snap
                if snap_and_check(U_final, V_final, W_final):
                    U_s = np.round(U_final * 2) / 2
                    V_s = np.round(V_final * 2) / 2
                    W_s = np.round(W_final * 2) / 2
                    return {
                        "U": U_s.tolist(),
                        "V": V_s.tolist(),
                        "W": W_s.tolist()
                    }
                
                # Try local search
                U_ls, V_ls, W_ls, loss_ls = local_search_halfint(U_final, V_final, W_final, max_iters=100)
                if loss_ls < 1e-9:
                    if snap_and_check(U_ls, V_ls, W_ls):
                        return {
                            "U": U_ls.tolist(),
                            "V": V_ls.tolist(),
                            "W": W_ls.tolist()
                        }
        
        if found:
            break

    return get_schoolbook()
