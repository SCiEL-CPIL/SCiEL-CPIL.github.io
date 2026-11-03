"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if U, V, W form a valid tensor decomposition."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 for all valid (i,j,k)
    
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    u_vals = Ua[u_idx, :]
    v_vals = Va[v_idx, :]
    w_vals = Wa[w_idx, :]
    
    prod = u_vals * v_vals * w_vals
    result = prod.sum(axis=1)
    
    expected = np.ones(n * m * p)
    
    return np.allclose(result, expected, atol=tol)

def _refine_gradient(U, V, W, n, m, p, rank, max_iter=200, tol=1e-10, lr_init=0.05):
    """Iteratively refine the decomposition using gradient descent."""
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    
    # Precompute indices
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Precompute index mappings for gradient accumulation
    # For V: v_idx maps (i,j,k) -> j*p+k
    # For W: w_idx maps (i,j,k) -> k*n+i
    
    lr = lr_init
    for it in range(max_iter):
        # Compute current values
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        error = result - 1.0
        err_norm = np.linalg.norm(error)
        
        if err_norm < tol:
            break
        
        # Compute gradients
        # L = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        # dL/dU[a,c] = 2 * sum_{(i,j,k): i*m+j=a} error[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
        
        err_3d = error.reshape(n, m, p)
        u_4d = u_vals.reshape(n, m, p, rank)
        v_4d = v_vals.reshape(n, m, p, rank)
        w_4d = w_vals.reshape(n, m, p, rank)
        
        # Gradient for U: sum over k
        grad_U_4d = 2.0 * np.expand_dims(err_3d, axis=-1) * v_4d * w_4d
        grad_U = np.sum(grad_U_4d, axis=2).reshape(n * m, rank)
        
        # Gradient for V: sum over i
        grad_V_4d = 2.0 * np.expand_dims(err_3d, axis=-1) * u_4d * w_4d
        # Need to sum over i for fixed (j,k)
        grad_V = np.zeros((m * p, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    grad_V[j * p + k, :] += grad_V_4d[i, j, k, :]
        
        # Gradient for W: sum over j
        grad_W_4d = 2.0 * np.expand_dims(err_3d, axis=-1) * u_4d * v_4d
        grad_W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    grad_W[k * n + i, :] += grad_W_4d[i, j, k, :]
        
        # Update with adaptive learning rate
        Ua -= lr * grad_U
        Va -= lr * grad_V
        Wa -= lr * grad_W
        
        # Decay learning rate
        lr *= 0.995
    
    return Ua.tolist(), Va.tolist(), Wa.tolist()

def _try_rank(rank, n, m, p, deadline, max_attempts=20):
    """Try to find a valid decomposition at the given rank."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        seed = rank * 10000 + attempt * 7 + 13
        random.seed(seed)
        np.random.seed(seed)
        
        # Random initialization
        scale = 1.0 / np.sqrt(rank)
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        # Try refinement
        U_ref, V_ref, W_ref = _refine_gradient(
            U.tolist(), V.tolist(), W.tolist(),
            n, m, p, rank,
            max_iter=100, tol=1e-10, lr_init=0.1
        )
        
        if _verify(U_ref, V_ref, W_ref, n, m, p, rank, tol=1e-8):
            return {"U": U_ref, "V": V_ref, "W": W_ref}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Start with schoolbook as guaranteed fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Known lower bounds for matrix multiplication:
    # For 4x4 * 4x7, the trivial bound is n*m*p = 112
    # Strassen-like methods might reduce this, but exact integer decompositions are rare
    # Let's search from a reasonable lower bound upward
    
    # Theoretically, rank >= max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But for exact decomposition, we likely need more
    # Let's try ranks from 28 to 112
    
    # Try some strategic ranks first
    candidate_ranks = []
    
    # Add some heuristically good ranks
    for r in range(28, best_rank + 1):
        candidate_ranks.append(r)
    
    # Also try some specific values that might work
    for r in [32, 40, 48, 56, 64, 72, 80, 88, 96, 100, 104, 108, 110, 111]:
        if 28 <= r <= best_rank and r not in candidate_ranks:
            candidate_ranks.append(r)
    
    candidate_ranks = sorted(set(candidate_ranks))
    
    # Try ranks in order
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            break
        
        result = _try_rank(rank, n, m, p, deadline, max_attempts=10)
        
        if result is not None:
            best = result
            best_rank = rank
            # Continue searching for even lower ranks
            # But we found one, so we can be more aggressive
            break
    
    # If we didn't find anything better, return schoolbook
    return best
# EVOLVE-BLOCK-END
