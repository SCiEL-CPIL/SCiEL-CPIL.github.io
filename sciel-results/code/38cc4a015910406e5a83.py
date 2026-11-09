"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
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

def _verify(decomp, n, m, p):
    """Verify that the tensor decomposition exactly multiplies the matrices."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    r = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for idx in range(r):
                    val += U[i*m + j][idx] * V[j*p + k][idx] * W[k*n + i][idx]
                if not (abs(val - 1.0) < 1e-9):
                    return False
    return True

def _compute_loss_and_grads(U, V, W, n, m, p):
    """Compute loss and gradients for the tensor decomposition using vectorized operations."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    rank = U.shape[1]
    
    # Precompute indices
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
    
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    # Vectorized computation
    U_sel = U[u_indices]  # (n*m*p, rank)
    V_sel = V[v_indices]  # (n*m*p, rank)
    W_sel = W[w_indices]  # (n*m*p, rank)
    
    # Compute values: sum over rank dimension
    values = np.sum(U_sel * V_sel * W_sel, axis=1)  # (n*m*p,)
    diffs = values - 1.0
    loss = np.sum(diffs * diffs)
    
    # Gradients
    factor = 2.0 * diffs  # (n*m*p,)
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    # Scatter gradients
    for idx in range(len(u_indices)):
        u_idx = u_indices[idx]
        v_idx = v_indices[idx]
        w_idx = w_indices[idx]
        f = factor[idx]
        grad_U[u_idx] += f * V_sel[idx] * W_sel[idx]
        grad_V[v_idx] += f * U_sel[idx] * W_sel[idx]
        grad_W[w_idx] += f * U_sel[idx] * V_sel[idx]
    
    return loss, grad_U, grad_V, grad_W

def _lbfgs_loss_grads(x, n, m, p, rank):
    """Flatten/unflatten for L-BFGS optimization."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    U = x[:num_u * rank].reshape(num_u, rank)
    V = x[num_u * rank:num_u * rank + num_v * rank].reshape(num_v, rank)
    W = x[num_u * rank + num_v * rank:].reshape(num_w, rank)
    
    loss, grad_U, grad_V, grad_W = _compute_loss_and_grads(U, V, W, n, m, p)
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    return loss, grad

def _polish(U, V, W, n, m, p, max_iter=100, tol=1e-14):
    """Final high-precision polish pass using L-BFGS."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    rank = U.shape[1]
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    best_loss = np.inf
    best_x = x0.copy()
    
    for it in range(max_iter):
        result = minimize(_lbfgs_loss_grads, x0, args=(n, m, p, rank), 
                         method='L-BFGS-B', jac=True, options={'maxiter': 50, 'ftol': 1e-16, 'gtol': 1e-14})
        
        if result.fun < best_loss:
            best_loss = result.fun
            best_x = result.x.copy()
        
        if best_loss < tol:
            break
        
        x0 = result.x
    
    U_opt = best_x[:num_u * rank].reshape(num_u, rank)
    V_opt = best_x[num_u * rank:num_u * rank + num_v * rank].reshape(num_v, rank)
    W_opt = best_x[num_u * rank + num_v * rank:].reshape(num_w, rank)
    
    return U_opt, V_opt, W_opt, best_loss

def _try_rank(n, m, p, rank, max_iter=300, seed=42):
    """Try to find a rank-r decomposition using L-BFGS optimization."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    best_loss = np.inf
    best_U, best_V, best_W = None, None, None
    
    # Try multiple initialization strategies
    for init_seed in range(5):
        init_rng = np.random.RandomState(seed + init_seed * 1000)
        
        # Different initialization strategies
        strategy = init_seed % 5
        
        if strategy == 0:
            # Small random initialization
            scale = 0.1
            U = init_rng.randn(num_u, rank) * scale
            V = init_rng.randn(num_v, rank) * scale
            W = init_rng.randn(num_w, rank) * scale
        elif strategy == 1:
            # Medium random initialization
            scale = 0.5
            U = init_rng.randn(num_u, rank) * scale
            V = init_rng.randn(num_v, rank) * scale
            W = init_rng.randn(num_w, rank) * scale
        elif strategy == 2:
            # Large random initialization
            scale = 1.0
            U = init_rng.randn(num_u, rank) * scale
            V = init_rng.randn(num_v, rank) * scale
            W = init_rng.randn(num_w, rank) * scale
        elif strategy == 3:
            # Sparse initialization - only a few non-zero entries per row
            U = np.zeros((num_u, rank))
            V = np.zeros((num_v, rank))
            W = np.zeros((num_w, rank))
            for r in range(rank):
                # Initialize with a few random 1s
                u_idx = init_rng.randint(0, num_u)
                v_idx = init_rng.randint(0, num_v)
                w_idx = init_rng.randint(0, num_w)
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
        else:
            # Structured initialization based on schoolbook pattern
            U = np.zeros((num_u, rank))
            V = np.zeros((num_v, rank))
            W = np.zeros((num_w, rank))
            for r in range(rank):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
        
        # Run L-BFGS optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        result = minimize(_lbfgs_loss_grads, x0, args=(n, m, p, rank), 
                         method='L-BFGS-B', jac=True, options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14})
        
        if result.fun < best_loss:
            best_loss = result.fun
            best_x = result.x.copy()
            U_opt = best_x[:num_u * rank].reshape(num_u, rank)
            V_opt = best_x[num_u * rank:num_u * rank + num_v * rank].reshape(num_v, rank)
            W_opt = best_x[num_u * rank + num_v * rank:].reshape(num_w, rank)
            best_U, best_V, best_W = U_opt, V_opt, W_opt
        
        if best_loss < 1e-14:
            break
    
    if best_U is not None and best_loss < 1e-14:
        # Final polish
        best_U, best_V, best_W, final_loss = _polish(best_U, best_V, best_W, n, m, p, max_iter=50)
        
        if final_loss < 1e-14:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist(), "rank": rank}
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    best_decomp = _schoolbook()
    best_rank = best_decomp["rank"]
    
    # Try ranks from known lower bound up to schoolbook
    # For 2x5x6, known lower bounds suggest rank around 20-30
    for rank in range(18, best_rank):
        if time.time() > deadline:
            break
        
        # More seeds for lower ranks where search is harder
        num_seeds = 8 if rank < 25 else 5
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, max_iter=200, seed=seed + 100)
            if result is not None:
                if _verify(result, n, m, p):
                    best_decomp = result
                    best_rank = rank
                    break
        if best_rank == rank:
            break
    
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
# EVOLVE-BLOCK-END
