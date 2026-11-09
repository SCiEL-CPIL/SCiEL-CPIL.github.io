"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

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
    U = np.array(decomp["U"])
    V = np.array(decomp["V"])
    W = np.array(decomp["W"])
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m + j] * V[j*p + k] * W[k*n + i])
                if not (abs(val - 1.0) < 1e-9):
                    return False
    return True

def _polish(U, V, W, n, m, p, max_iter=200, tol=1e-15):
    """High-precision polish pass with adaptive learning rate."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        loss = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    diff = val - 1.0
                    loss += diff * diff
                    
                    factor = 2.0 * diff
                    grad_U[u_idx] += factor * V[v_idx] * W[w_idx]
                    grad_V[v_idx] += factor * U[u_idx] * W[w_idx]
                    grad_W[w_idx] += factor * U[u_idx] * V[v_idx]
        
        # Adaptive learning rate
        norm_U = np.linalg.norm(grad_U)
        norm_V = np.linalg.norm(grad_V)
        norm_W = np.linalg.norm(grad_W)
        max_norm = max(norm_U, norm_V, norm_W, 1e-10)
        effective_lr = lr * (1.0 / max_norm)
        
        U_new = U - effective_lr * grad_U
        V_new = V - effective_lr * grad_V
        W_new = W - effective_lr * grad_W
        
        # Check if loss improved
        new_loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_new[i*m + j] * V_new[j*p + k] * W_new[k*n + i])
                    new_loss += (val - 1.0) ** 2
        
        if new_loss < loss * 0.999:
            U, V, W = U_new, V_new, W_new
            loss = new_loss
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if loss < tol:
                break
        else:
            lr *= 0.5
    
    return best_U, best_V, best_W, best_loss

def _try_rank_lbfgs(n, m, p, rank, seed=42, maxiter=500):
    """Try to find a rank-r decomposition using L-BFGS-B optimization."""
    from scipy.optimize import minimize
    
    rng = np.random.RandomState(seed)
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Multiple initialization strategies
    inits = []
    
    # Random Gaussian
    U_init = rng.randn(num_u, rank) * 0.1
    V_init = rng.randn(num_v, rank) * 0.1
    W_init = rng.randn(num_w, rank) * 0.1
    inits.append((U_init, V_init, W_init))
    
    # Sparse random
    U_init = rng.rand(num_u, rank) * (rng.rand(num_u, rank) < 0.3)
    V_init = rng.rand(num_v, rank) * (rng.rand(num_v, rank) < 0.3)
    W_init = rng.rand(num_w, rank) * (rng.rand(num_w, rank) < 0.3)
    inits.append((U_init, V_init, W_init))
    
    # Polynomial-like initialization
    U_init = np.zeros((num_u, rank))
    V_init = np.zeros((num_v, rank))
    W_init = np.zeros((num_w, rank))
    for idx in range(rank):
        U_init[:, idx] = rng.randn(num_u) * 0.05
        V_init[:, idx] = rng.randn(num_v) * 0.05
        W_init[:, idx] = rng.randn(num_w) * 0.05
    inits.append((U_init, V_init, W_init))
    
    for init_idx, (U0, V0, W0) in enumerate(inits):
        # Flatten parameters
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def loss_and_grads(x):
            U = x[:num_u*rank].reshape(num_u, rank)
            V = x[num_u*rank:num_u*rank+num_v*rank].reshape(num_v, rank)
            W = x[num_u*rank+num_v*rank:].reshape(num_w, rank)
            
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
                        
                        val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                        diff = val - 1.0
                        loss += diff * diff
                        
                        factor = 2.0 * diff
                        grad_U[u_idx] += factor * V[v_idx] * W[w_idx]
                        grad_V[v_idx] += factor * U[u_idx] * W[w_idx]
                        grad_W[w_idx] += factor * U[u_idx] * V[v_idx]
            
            grads = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grads
        
        result = minimize(loss_and_grads, x0, method='L-BFGS-B', 
                         jac=True, options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12})
        
        if result.success or result.fun < 1e-10:
            x_opt = result.x
            U_opt = x_opt[:num_u*rank].reshape(num_u, rank)
            V_opt = x_opt[num_u*rank:num_u*rank+num_v*rank].reshape(num_v, rank)
            W_opt = x_opt[num_u*rank+num_v*rank:].reshape(num_w, rank)
            
            # Polish pass
            U_pol, V_pol, W_pol, final_loss = _polish(U_opt, V_opt, W_opt, n, m, p)
            
            if final_loss < 1e-12:
                decomp = {"U": U_pol.tolist(), "V": V_pol.tolist(), "W": W_pol.tolist(), "rank": rank}
                if _verify(decomp, n, m, p):
                    return decomp
    
    return None

def _try_rank_gradient(n, m, p, rank, max_iter=1000, seed=42):
    """Try to find a rank-r decomposition using improved gradient descent."""
    rng = np.random.RandomState(seed)
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Initialize with small random values
    U = rng.randn(num_u, rank) * 0.05
    V = rng.randn(num_v, rank) * 0.05
    W = rng.randn(num_w, rank) * 0.05
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        loss = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    diff = val - 1.0
                    loss += diff * diff
                    
                    factor = 2.0 * diff
                    grad_U[u_idx] += factor * V[v_idx] * W[w_idx]
                    grad_V[v_idx] += factor * U[u_idx] * W[w_idx]
                    grad_W[w_idx] += factor * U[u_idx] * V[v_idx]
        
        # Adaptive learning rate
        norm_U = np.linalg.norm(grad_U)
        norm_V = np.linalg.norm(grad_V)
        norm_W = np.linalg.norm(grad_W)
        max_norm = max(norm_U, norm_V, norm_W, 1e-10)
        effective_lr = lr / max_norm
        
        U -= effective_lr * grad_U
        V -= effective_lr * grad_V
        W -= effective_lr * grad_W
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-14:
            break
        
        # Reduce learning rate if stuck
        if it > 100 and loss > best_loss * 1.01:
            lr *= 0.5
    
    if best_loss < 1e-12:
        U_pol, V_pol, W_pol, final_loss = _polish(best_U, best_V, best_W, n, m, p)
        if final_loss < 1e-12:
            decomp = {"U": U_pol.tolist(), "V": V_pol.tolist(), "W": W_pol.tolist(), "rank": rank}
            if _verify(decomp, n, m, p):
                return decomp
    
    return None

def construct():
    start_time = time.time()
    deadline = 148.0
    
    n, m, p = DIMENSIONS
    best_decomp = _schoolbook()
    best_rank = best_decomp["rank"]
    
    # Try ranks from low to high, with multiple methods
    for rank in range(35, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        
        # Try L-BFGS first (usually faster convergence)
        for seed in range(2):
            if time.time() > deadline:
                break
            result = _try_rank_lbfgs(n, m, p, rank, seed=seed + 200, maxiter=300)
            if result is not None:
                best_decomp = result
                best_rank = rank
                found = True
                break
        
        if found:
            break
        
        # Try gradient descent as fallback
        for seed in range(3):
            if time.time() > deadline:
                break
            result = _try_rank_gradient(n, m, p, rank, max_iter=500, seed=seed + 500)
            if result is not None:
                best_decomp = result
                best_rank = rank
                found = True
                break
    
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
# EVOLVE-BLOCK-END
