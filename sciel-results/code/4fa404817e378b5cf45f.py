"""Matrix multiplication 3x3x5: tensor decomposition with deadline guard."""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _compute_tensor_error(U, V, W, n, m, p):
    """Compute the error between the decomposition and the true multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    error = 0.0
    
    # Check consistent entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                error += (val - 1.0) ** 2
    
    # Check inconsistent entries (should be 0)
    for u in range(n * m):
        i_u = u // m
        j_u = u % m
        for v in range(m * p):
            j_v = v // p
            k_v = v % p
            for w in range(p * n):
                k_w = w // n
                i_w = w % n
                # Skip consistent entries
                if j_u == j_v and i_u == i_w and k_v == k_w:
                    continue
                val = np.sum(U[u, :] * V[v, :] * W[w, :])
                error += val ** 2
    
    return error

def _try_structured_decomposition(rank, n, m, p, max_iterations=5000, lr=0.01, seed=42):
    """Try to find a rank decomposition using structured initialization and gradient descent."""
    rng = np.random.RandomState(seed)
    best_error = float('inf')
    best_params = None
    
    # Initialize with a structured pattern
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Try different initialization strategies
    for init_strategy in range(3):
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        if init_strategy == 0:
            # Random small values
            U_curr = rng.randn(n * m, rank) * 0.1
            V_curr = rng.randn(m * p, rank) * 0.1
            W_curr = rng.randn(p * n, rank) * 0.1
        elif init_strategy == 1:
            # Sparse random
            U_curr = (rng.rand(n * m, rank) < 0.3) * rng.randn(n * m, rank) * 0.5
            V_curr = (rng.rand(m * p, rank) < 0.3) * rng.randn(m * p, rank) * 0.5
            W_curr = (rng.rand(p * n, rank) < 0.3) * rng.randn(p * n, rank) * 0.5
        else:
            # Block-structured based on indices
            for r in range(rank):
                i = r // (m * p)
                j = (r // p) % m
                k = r % p
                if i < n:
                    U_curr[i * m + j, r] = 1.0
                V_curr[j * p + k, r] = 1.0
                W_curr[k * n + i, r] = 1.0
        
        # Gradient descent
        for iteration in range(max_iterations):
            error = _compute_tensor_error(U_curr, V_curr, W_curr, n, m, p)
            if error < best_error:
                best_error = error
                best_params = (U_curr.copy(), V_curr.copy(), W_curr.copy())
            
            if error < 1e-10:
                break
            
            # Compute gradients numerically (expensive but reliable)
            eps = 1e-5
            grad_U = np.zeros_like(U_curr)
            grad_V = np.zeros_like(V_curr)
            grad_W = np.zeros_like(W_curr)
            
            # Sample a subset of entries for gradient estimation
            num_samples = 100
            u_indices = rng.randint(0, n * m, num_samples)
            v_indices = rng.randint(0, m * p, num_samples)
            w_indices = rng.randint(0, p * n, num_samples)
            
            for idx in range(num_samples):
                u, v, w = u_indices[idx], v_indices[idx], w_indices[idx]
                
                i_u, j_u = divmod(u, m)
                j_v, k_v = divmod(v, p)
                k_w, i_w = divmod(w, n)
                
                # Determine if this is a consistent entry
                is_consistent = (j_u == j_v and i_u == i_w and k_v == k_w)
                target = 1.0 if is_consistent else 0.0
                
                # Current value
                val = np.sum(U_curr[u, :] * V_curr[v, :] * W_curr[w, :])
                diff = val - target
                
                # Gradient with respect to U[u, r]
                for r in range(rank):
                    grad_U[u, r] += diff * V_curr[v, r] * W_curr[w, r]
                    grad_V[v, r] += diff * U_curr[u, r] * W_curr[w, r]
                    grad_W[w, r] += diff * U_curr[u, r] * V_curr[v, r]
            
            # Update
            U_curr -= lr * grad_U
            V_curr -= lr * grad_V
            W_curr -= lr * grad_W
        
        if best_error < 1e-10:
            break
    
    if best_error < 1e-10 and best_params is not None:
        U_best, V_best, W_best = best_params
        return U_best.tolist(), V_best.tolist(), W_best.tolist()
    
    return None

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank-45 decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return U, V, W

def construct():
    """Return a rank-minimal exact tensor decomposition for 3x3 * 3x5."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 30:
            break
        
        # Try structured search for this rank
        result = _try_structured_decomposition(rank, n, m, p, max_iterations=2000, lr=0.05, seed=rank)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found a valid decomposition
    
    return best_result
# EVOLVE-BLOCK-END
