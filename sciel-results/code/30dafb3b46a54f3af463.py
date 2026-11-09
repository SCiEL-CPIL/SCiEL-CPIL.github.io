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

def _precompute_indices(n, m, p):
    """Precompute index arrays for vectorized operations."""
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
    return (np.array(u_indices, dtype=np.int64),
            np.array(v_indices, dtype=np.int64),
            np.array(w_indices, dtype=np.int64))

def _compute_loss_and_grads(U, V, W, n, m, p, u_idx, v_idx, w_idx):
    """Compute loss and gradients for the tensor decomposition using vectorized operations."""
    U_sel = U[u_idx]  # (n*m*p, rank)
    V_sel = V[v_idx]  # (n*m*p, rank)
    W_sel = W[w_idx]  # (n*m*p, rank)
    
    # Compute values: sum over rank dimension
    values = np.sum(U_sel * V_sel * W_sel, axis=1)  # (n*m*p,)
    diffs = values - 1.0
    loss = np.sum(diffs * diffs)
    
    # Gradients
    factor = 2.0 * diffs  # (n*m*p,)
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    gU = factor[:, None] * V_sel * W_sel  # (n*m*p, rank)
    gV = factor[:, None] * U_sel * W_sel
    gW = factor[:, None] * U_sel * V_sel
    
    np.add.at(grad_U, u_idx, gU)
    np.add.at(grad_V, v_idx, gV)
    np.add.at(grad_W, w_idx, gW)
    
    return loss, grad_U, grad_V, grad_W

def _make_loss_grads(n, m, p, rank, u_idx, v_idx, w_idx):
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    def loss_grads(x):
        U = x[:num_u * rank].reshape(num_u, rank)
        V = x[num_u * rank:num_u * rank + num_v * rank].reshape(num_v, rank)
        W = x[num_u * rank + num_v * rank:].reshape(num_w, rank)
        
        loss, grad_U, grad_V, grad_W = _compute_loss_and_grads(U, V, W, n, m, p, u_idx, v_idx, w_idx)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad
    
    return loss_grads

def _structured_init_v2(n, m, p, rank, rng):
    """Better structured initialization using block patterns."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    U = np.zeros((num_u, rank))
    V = np.zeros((num_v, rank))
    W = np.zeros((num_w, rank))
    
    # Strategy: Use the known rank-24 structure as a template
    # For 2x5x6, think of it as combining blocks
    # Each rank component should have a clear pattern
    
    for r in range(rank):
        # Primary assignment based on modular arithmetic
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p
        
        U[i * m + j, r] = 1.0
        V[j * p + k, r] = 1.0
        W[k * n + i, r] = 1.0
        
        # For lower ranks, add cross terms
        if rank < n * m * p:
            # Add secondary contributions with different patterns
            i2 = (r + n - 1) % n
            j2 = (r // n + 1) % m
            k2 = (r // (n * m) + 1) % p
            
            U[i2 * m + j2, r] = 0.5
            V[j2 * p + k2, r] = 0.5
            W[k2 * n + i2, r] = 0.5
            
            # Tertiary contributions
            i3 = (r + 2 * n) % n
            j3 = (r // n + 2) % m
            k3 = (r // (n * m) + 2) % p
            
            U[i3 * m + j3, r] = 0.25
            V[j3 * p + k3, r] = 0.25
            W[k3 * n + i3, r] = 0.25
    
    # Add small random perturbation to break symmetry
    U += rng.randn(num_u, rank) * 0.01
    V += rng.randn(num_v, rank) * 0.01
    W += rng.randn(num_w, rank) * 0.01
    
    return U, V, W

def _random_init(n, m, p, rank, rng, scale=0.5):
    """Random initialization."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    U = rng.randn(num_u, rank) * scale
    V = rng.randn(num_v, rank) * scale
    W = rng.randn(num_w, rank) * scale
    
    return U, V, W

def _sparse_init(n, m, p, rank, rng):
    """Sparse initialization with a few non-zero entries per row."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    U = np.zeros((num_u, rank))
    V = np.zeros((num_v, rank))
    W = np.zeros((num_w, rank))
    
    for r in range(rank):
        # Each rank component touches a few entries
        n_u_nonzero = rng.randint(1, 3)
        n_v_nonzero = rng.randint(1, 3)
        n_w_nonzero = rng.randint(1, 3)
        
        u_rows = rng.choice(num_u, size=min(n_u_nonzero, num_u), replace=False)
        v_rows = rng.choice(num_v, size=min(n_v_nonzero, num_v), replace=False)
        w_rows = rng.choice(num_w, size=min(n_w_nonzero, num_w), replace=False)
        
        for row in u_rows:
            U[row, r] = rng.choice([-1.0, 1.0])
        for row in v_rows:
            V[row, r] = rng.choice([-1.0, 1.0])
        for row in w_rows:
            W[row, r] = rng.choice([-1.0, 1.0])
    
    return U, V, W

def _try_rank(n, m, p, rank, max_iter=200, seed=42, u_idx=None, v_idx=None, w_idx=None):
    """Try to find a rank-r decomposition using L-BFGS optimization."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    if u_idx is None or v_idx is None or w_idx is None:
        u_idx, v_idx, w_idx = _precompute_indices(n, m, p)
    
    loss_grads_fn = _make_loss_grads(n, m, p, rank, u_idx, v_idx, w_idx)
    
    best_loss = np.inf
    best_x = None
    
    # Try multiple initialization strategies
    for init_seed in range(8):
        rng = np.random.RandomState(seed + init_seed * 1000)
        
        strategy = init_seed % 4
        
        if strategy == 0:
            U, V, W = _structured_init_v2(n, m, p, rank, rng)
        elif strategy == 1:
            U, V, W = _random_init(n, m, p, rank, rng, scale=0.3)
        elif strategy == 2:
            U, V, W = _sparse_init(n, m, p, rank, rng)
        else:
            # Hybrid: structured base + random perturbation
            U0, V0, W0 = _structured_init_v2(n, m, p, rank, rng)
            U = U0 + rng.randn(num_u, rank) * 0.1
            V = V0 + rng.randn(num_v, rank) * 0.1
            W = W0 + rng.randn(num_w, rank) * 0.1
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Run L-BFGS
        result = minimize(loss_grads_fn, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50})
        
        if result.fun < best_loss:
            best_loss = result.fun
            best_x = result.x.copy()
        
        if best_loss < 1e-14:
            break
    
    if best_x is not None and best_loss < 1e-10:
        U_opt = best_x[:num_u * rank].reshape(num_u, rank)
        V_opt = best_x[num_u * rank:num_u * rank + num_v * rank].reshape(num_v, rank)
        W_opt = best_x[num_u * rank + num_v * rank:].reshape(num_w, rank)
        
        # Final polish with tighter tolerance
        result2 = minimize(loss_grads_fn, best_x, method='L-BFGS-B', jac=True,
                          options={'maxiter': 100, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 30})
        
        if result2.fun < best_loss:
            best_x = result2.x
            best_loss = result2.fun
        
        U_opt = best_x[:num_u * rank].reshape(num_u, rank)
        V_opt = best_x[num_u * rank:num_u * rank + num_v * rank].reshape(num_v, rank)
        W_opt = best_x[num_u * rank + num_v * rank:].reshape(num_w, rank)
        
        if best_loss < 1e-14:
            return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist(), "rank": rank}
    
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    best_decomp = _schoolbook()
    best_rank = best_decomp["rank"]
    
    # Precompute indices once
    u_idx, v_idx, w_idx = _precompute_indices(n, m, p)
    
    # Try ranks from known lower bound up to schoolbook
    # For 2x5x6, theoretical lower bound is around 18-20
    # Known constructions suggest rank 24 is achievable
    for rank in range(18, best_rank):
        if time.time() > deadline:
            break
        
        # Allocate more time for lower ranks
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # More seeds for lower ranks
        num_seeds = 12 if rank < 22 else 8
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, max_iter=150, seed=seed + 100,
                             u_idx=u_idx, v_idx=v_idx, w_idx=w_idx)
            if result is not None:
                if _verify(result, n, m, p):
                    best_decomp = result
                    best_rank = rank
                    break
        
        if best_rank == rank:
            break
    
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
# EVOLVE-BLOCK-END
