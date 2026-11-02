"""Find rank-minimal exact tensor decomposition for 2x3 by 3x4 matrix multiplication."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_tensor(n, m, p):
    return np.ones((n, m, p))

def _reconstruct(U, V, W, n, m, p, rank):
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return T

def _validate(U, V, W, n, m, p, rank, tol=1e-9):
    T = _reconstruct(U, V, W, n, m, p, rank)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _snap_to_half_integers(U, V, W, tol=0.01):
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0
    if (np.max(np.abs(U - U_s)) < tol and
        np.max(np.abs(V - V_s)) < tol and
        np.max(np.abs(W - W_s)) < tol):
        U_s = np.round(U_s * 2) / 2.0
        V_s = np.round(V_s * 2) / 2.0
        W_s = np.round(W_s * 2) / 2.0
        return U_s, V_s, W_s
    return None

def _objective_and_grad(params, n, m, p, rank, T_target):
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    
    U = params[:U_size].reshape(n, m, rank)
    V = params[U_size:U_size+V_size].reshape(m, p, rank)
    W = params[U_size+V_size:].reshape(p, n, rank)
    
    T = _reconstruct(U.flatten(), V.flatten(), W.flatten(), n, m, p, rank)
    residual = T - T_target
    obj = np.sum(residual**2)
    
    dT = 2 * residual
    dU_r = np.einsum('ijk,jkc,kic->ijc', dT, V, W)
    dV_r = np.einsum('ijk,ijc,kic->jkc', dT, U, W)
    dW_r = np.einsum('ijk,ijc,jkc->kic', dT, U, V)
    
    grad = np.concatenate([dU_r.flatten(), dV_r.flatten(), dW_r.flatten()])
    return obj, grad

def _try_optimization(n, m, p, target_rank, seed, time_limit, max_iter=10000):
    rng = np.random.RandomState(seed)
    T_target = _build_tensor(n, m, p)
    start_time = time.time()
    
    best_obj = float('inf')
    best_params = None
    
    n_init = 5
    for init_i in range(n_init):
        if time.time() - start_time > time_limit - 5:
            break
        
        scales = [0.1, 0.5, 1.0, 2.0, 0.3]
        scale = scales[init_i % len(scales)]
        
        U = rng.randn(n, m, target_rank) * scale
        V = rng.randn(m, p, target_rank) * scale
        W = rng.randn(p, n, target_rank) * scale
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(
                lambda x: _objective_and_grad(x, n, m, p, target_rank, T_target)[0],
                params,
                jac=lambda x: _objective_and_grad(x, n, m, p, target_rank, T_target)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-13}
            )
        except Exception:
            continue
        
        obj = result.fun
        if obj < best_obj:
            best_obj = obj
            best_params = result.x
        
        if best_obj < 1e-12:
            break
    
    if best_params is not None and best_obj < 1e-9:
        U_size = n * m * target_rank
        V_size = m * p * target_rank
        
        U = best_params[:U_size].reshape(n, m, target_rank).flatten()
        V = best_params[U_size:U_size+V_size].reshape(m, p, target_rank).flatten()
        W = best_params[U_size+V_size:].reshape(p, n, target_rank).flatten()
        
        snapped = _snap_to_half_integers(U, V, W)
        if snapped is not None:
            U_s, V_s, W_s = snapped
            if _validate(U_s, V_s, W_s, n, m, p, target_rank):
                return U_s, V_s, W_s
        
        if _validate(U, V, W, n, m, p, target_rank):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    time_limit = 140
    
    for rank in range(13, schoolbook_rank):
        if time.time() - start_time > time_limit - 10:
            break
        
        rank_time = min(35, time_limit - (time.time() - start_time))
        if rank_time < 10:
            break
        
        result = _try_optimization(n, m, p, rank, seed=rank * 100 + 7, time_limit=rank_time)
        if result is not None:
            U_s, V_s, W_s = result
            U_list = [[float(val) for val in row] for row in U_s.tolist()]
            V_list = [[float(val) for val in row] for row in V_s.tolist()]
            W_list = [[float(val) for val in row] for row in W_s.tolist()]
            
            if _validate(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
