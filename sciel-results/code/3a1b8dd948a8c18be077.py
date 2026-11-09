"""Tensor decomposition for 2x5x6 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify(decomp, n, m, p, tol=1e-9):
    U = np.array(decomp["U"])
    V = np.array(decomp["V"])
    W = np.array(decomp["W"])
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m + j] * V[j*p + k] * W[k*n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _loss_and_grads(params, n, m, p, rank):
    """Compute loss and gradient for tensor decomposition."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    U = params[0:num_u*rank].reshape(num_u, rank)
    V = params[num_u*rank:num_u*rank+num_v*rank].reshape(num_v, rank)
    W = params[num_u*rank+num_v*rank:].reshape(num_w, rank)
    
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                val = np.dot(U[u_idx], np.multiply(V[v_idx], W[w_idx]))
                diff = val - 1.0
                loss += diff * diff
                
                factor = 2.0 * diff
                grad_U[u_idx] += factor * V[v_idx] * W[w_idx]
                grad_V[v_idx] += factor * U[u_idx] * W[w_idx]
                grad_W[w_idx] += factor * U[u_idx] * V[v_idx]
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    return loss, grad

def _try_rank(n, m, p, rank, seed=42, max_iter=2000):
    """Try to find a rank-r decomposition using L-BFGS."""
    rng = np.random.RandomState(seed)
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    best_loss = np.inf
    best_params = None
    
    for init_seed in range(5):
        init_rng = np.random.RandomState(seed + init_seed * 1000)
        scale = 0.5
        U = init_rng.randn(num_u, rank) * scale
        V = init_rng.randn(num_v, rank) * scale
        W = init_rng.randn(num_w, rank) * scale
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        res = minimize(
            _loss_and_grads,
            params0,
            args=(n, m, p, rank),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
        
        if best_loss < 1e-14:
            break
    
    if best_params is not None and best_loss < 1e-14:
        U = best_params[0:num_u*rank].reshape(num_u, rank)
        V = best_params[num_u*rank:num_u*rank+num_v*rank].reshape(num_v, rank)
        W = best_params[num_u*rank+num_v*rank:].reshape(num_w, rank)
        
        decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": rank}
        if _verify(decomp, n, m, p):
            return decomp
    
    return None

def _try_rank_gks(n, m, p, rank, seed=42):
    """Try GKS-inspired initialization then optimize."""
    rng = np.random.RandomState(seed)
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Initialize with a structured pattern
    U = np.zeros((num_u, rank))
    V = np.zeros((num_v, rank))
    W = np.zeros((num_w, rank))
    
    # Use a sparse initialization
    for alpha in range(rank):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        U[i*m+j, alpha] = 1.0
        V[j*p+k, alpha] = 1.0
        W[k*n+i, alpha] = 1.0
    
    # Add some randomness
    U += rng.randn(num_u, rank) * 0.1
    V += rng.randn(num_v, rank) * 0.1
    W += rng.randn(num_w, rank) * 0.1
    
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    res = minimize(
        _loss_and_grads,
        params0,
        args=(n, m, p, rank),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
    )
    
    if res.fun < 1e-14:
        U = res.x[0:num_u*rank].reshape(num_u, rank)
        V = res.x[num_u*rank:num_u*rank+num_v*rank].reshape(num_v, rank)
        W = res.x[num_u*rank+num_v*rank:].reshape(num_w, rank)
        
        decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": rank}
        if _verify(decomp, n, m, p):
            return decomp
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    best_decomp = _schoolbook()
    best_rank = best_decomp["rank"]
    
    # Try ranks from low to high
    for rank in range(25, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple seeds
        for seed in range(3):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, seed=seed, max_iter=2000)
            if result is not None:
                best_decomp = result
                best_rank = rank
                found = True
                break
        
        if not found and rank < 50:
            # Also try GKS-inspired
            for seed in range(2):
                if time.time() > deadline:
                    break
                result = _try_rank_gks(n, m, p, rank, seed=seed)
                if result is not None:
                    best_decomp = result
                    best_rank = rank
                    found = True
                    break
        
        if best_rank == rank:
            break
    
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
